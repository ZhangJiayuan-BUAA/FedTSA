import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from sklearn.cluster import KMeans
import copy

class TempoAttention(nn.Module):
    def __init__(self, dim, heads, dropout=.1):
        super(TempoAttention, self).__init__()
        self.fc_q = nn.Linear(dim, heads * dim)
        self.fc_k = nn.Linear(dim, heads * dim)
        self.fc_v = nn.Linear(dim, heads * dim).requires_grad_(False)
        self.dropout = nn.Dropout(dropout)

        self.dim = dim
        self.heads = heads
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.eye_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x, attention_mask=None, attention_weights=None):
        N, T, D = x.shape   # X^{N T D}

        q = self.fc_q(x).view(N, T, self.heads, D).permute(0, 2, 1, 3)
        k = self.fc_k(x).view(N, T, self.heads, D).permute(0, 2, 3, 1)
        v = self.fc_v(x).view(N, T, self.heads, D).permute(0, 2, 1, 3)

        self.att = torch.matmul(q, k) / np.sqrt(D)  # (N, heads, T, T)
        if attention_weights is not None:
            self.att = self.att * attention_weights
        if attention_mask is not None:
            self.att = self.att.masked_fill(attention_mask, -np.inf)
        self.att = torch.softmax(self.att, -1)

        out = torch.matmul(self.att, v).permute(0, 2, 1, 3).contiguous().view(N, T, self.heads * D)  # (N, T, h*d_v)
        out_agg = torch.mean(out, dim=1)

        return out, out_agg.detach(), self.att.detach()

def graphStructual(avg_feature, sim_type='cos', threshold=0.9):
    N, D = avg_feature.shape
    adj_matrix = torch.matmul(avg_feature, avg_feature.t())
    if sim_type == 'cos':
        scaling = torch.outer(torch.norm(avg_feature, p=2, dim=1), torch.norm(avg_feature, p=2, dim=1)) ** -1
        sim = adj_matrix * scaling
    elif sim_type == 'att':
        scaling = float(D) ** -0.5
        sim = torch.softmax(adj_matrix * scaling, dim=-1)
    else:
        raise ValueError('graphStructual only support [cos, att]')
    threshold = torch.mean(sim)
    graph = (sim > threshold).to(torch.int)
    return sim, graph

class SpatialAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0., qkv_bias=True):
        super(SpatialAttention, self).__init__()
        self.geo_q = nn.Linear(dim, heads * dim, bias=qkv_bias)
        self.geo_k = nn.Linear(dim, heads * dim, bias=qkv_bias)
        self.geo_v = nn.Linear(dim, heads * dim, bias=qkv_bias).requires_grad_(False)
        self.dropout = nn.Dropout(dropout)

        self.dim = dim
        self.heads = heads

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.eye_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x, attention_mask=None, attention_weights=None):
        N, T, D = x.shape

        q = self.geo_q(x).view(N, T, self.heads, D).permute(1, 2, 0, 3)  # (T, head, N, D)
        k = self.geo_k(x).view(N, T, self.heads, D).permute(1, 2, 3, 0)  # (T, head, D, N)
        v = self.geo_v(x).view(N, T, self.heads, D).permute(1, 2, 0, 3)   # (T, head, N, D)

        self.att = torch.matmul(q, k) / np.sqrt(D)  # (T, head, D, D)
        raw_att = self.att
        raw_att = torch.softmax(raw_att, dim=-1)
        out_raw = torch.matmul(raw_att, v).permute(1, 2, 0, 3).contiguous().view(N, T, self.heads * D)  # (b_s, nq, h*d_v)

        if attention_weights is not None:
            self.att = self.att * attention_weights
        if attention_mask is not None:
            self.att = self.att.masked_fill(attention_mask, -np.inf)
        self.att = torch.softmax(self.att, dim=-1)

        out = torch.matmul(self.att, v).permute(1, 2, 0, 3).contiguous().view(N, T, self.heads * D)  # (b_s, nq, h*d_v)

        return out, out_raw, raw_att[-1, 0, :, :].clone().detach()

class ST_block(nn.Module):
    def __init__(self, args, dim, Theads=1, Sheads=1, proj_bias=False, proj_dropout=.0):
        super().__init__()
        self.dim = dim
        self.args = args
        Theads = args.other.st_head
        Sheads = args.other.st_head
        self.TA1 = TempoAttention(dim=dim, heads=Theads)
        self.SA1 = SpatialAttention(dim=dim, heads=Sheads)
        # self.TA2 = TempoAttention(dim=dim, heads=Theads)

    def forward(self, x, wotime=False):
        if wotime:
            x1, x_agg, t_sim_A = self.TA1(x) # x: after message passing between sequential batches, x_agg: mean which can indicate this area, t_sim_A: [ cidx ][ heads ][ T ]

            _, graph = graphStructual(x[:, -1, :])
            repr1, repr0, s_sim_A = self.SA1(x, attention_mask=(graph == 0))
            space_mask = aug_spatiol(s_sim_A, graph)
            repr2, _, _ = self.SA1(x, attention_mask=(space_mask == 0))

            return repr0, repr1, repr2, t_sim_A, s_sim_A
        else:
            x1, x_agg, t_sim_A = self.TA1(x)  # x: after message passing between sequential batches, x_agg: mean which can indicate this area, t_sim_A: [ cidx ][ heads ][ T ]
            # _, graph = graphStructual(x1[:, -1, :])
            if self.args.other.st == 'independent':
                _, graph = graphStructual(x[:, -1, :])
                repr1, repr0, s_sim_A = self.SA1(x, attention_mask=(graph == 0))

                space_mask = aug_spatiol(s_sim_A, graph)
                repr2, _, _ = self.SA1(x, attention_mask=(space_mask == 0))

            elif self.args.other.st == 'graph':
                _, graph = graphStructual(x1[:, -1, :])
                repr1, repr0, s_sim_A = self.SA1(x, attention_mask=(graph == 0))

                space_mask = aug_spatiol(s_sim_A, graph)
                repr2, _, _ = self.SA1(x, attention_mask=(space_mask == 0))

            elif self.args.other.st == 'seq':
                _, graph = graphStructual(x[:, -1, :])
                repr1, repr0, s_sim_A = self.SA1(x1, attention_mask=(graph == 0))
                time_mask = aug_temporal(t_sim_A)
                space_mask = aug_spatiol(s_sim_A, graph)
                x2, _, _ = self.TA1(x, attention_mask=(time_mask == 0))
                repr2, _, _ = self.SA1(x2, attention_mask=(space_mask == 0))

            else:
                _, graph = graphStructual(x1[:, -1, :])
                repr1, repr0, s_sim_A = self.SA1(x1, attention_mask=(graph == 0))
                if self.args.other.st == 'random':
                    time_mask = aug_temporal(t_sim_A, random=True)
                    space_mask = aug_spatiol(s_sim_A, graph, random=True)
                else:
                    time_mask = aug_temporal(t_sim_A)
                    space_mask = aug_spatiol(s_sim_A, graph)
                x2, _, _ = self.TA1(x, attention_mask=(time_mask == 0))
                repr2, _, _ = self.SA1(x2, attention_mask=(space_mask == 0))

            return repr0, repr1, repr2, t_sim_A, s_sim_A

def aug_temporal(t_sim_A, percent=0.75, random=False):
    N, heads, T, T = t_sim_A.shape
    mask_prob = (1. - t_sim_A).cpu().numpy()

    time_mask = torch.ones_like(t_sim_A)
    y, x = np.meshgrid(range(T), range(T))
    mask_number = int((T * T) * percent)
    zeros = torch.zeros_like(t_sim_A[0, 0, 0, 0])
    ones = torch.ones_like(t_sim_A[0, 0, 0, 0])
    for i in range(N):
        for j in range(heads):
            if mask_prob[i][j].sum() == 0:
                break
            mask_prob[i][j] /= mask_prob[i][j].sum()
            if len(mask_prob[i][j].reshape(-1)>0) > mask_number:
                if not random:
                    mask_list = np.random.choice(T * T, size=mask_number, p=mask_prob[i][j].reshape(-1), replace=False)
                else:
                    mask_list = np.random.choice(T * T, size=mask_number, replace=False)
                time_mask[i][j][
                    x.reshape(-1)[mask_list],
                    y.reshape(-1)[mask_list]
                ] = zeros
            for k in range(T):
                time_mask[i][j][k][k] = ones
    return time_mask

def aug_spatiol(sim_mx, graph, percent=0.8, random=False):
    drop_percent = percent
    add_percent = 1 - percent

    # mask some attention
    I = torch.eye(graph.shape[0]).cuda()
    input_graph = graph - I
    index_list = input_graph.nonzero()  # list of edges [row_idx, col_idx]

    # edge_num = int(index_list.shape[0] / 2)  # treat one undirected edge as two edges
    edge_num = int(index_list.shape[0])
    # edge_mask = (input_graph > 0).tril(diagonal=-1).cpu()
    edge_mask = (input_graph > 0).cpu()
    add_drop_num = int(edge_num * drop_percent)
    aug_graph = copy.deepcopy(input_graph)

    drop_prob = sim_mx[edge_mask]
    drop_prob = (1. - drop_prob).cpu().numpy()  # normalized similarity to get sampling probability
    drop_prob /= drop_prob.sum()
    if not random:
        drop_list = np.random.choice(edge_num, size=add_drop_num, p=drop_prob, replace=False)
    else:
        drop_list = np.random.choice(edge_num, size=add_drop_num, replace=False)
    drop_index = index_list[drop_list]

    zeros = torch.zeros_like(aug_graph[0, 0])
    aug_graph[drop_index[:, 0], drop_index[:, 1]] = zeros
    # aug_graph[drop_index[:, 1], drop_index[:, 0]] = zeros

    # add some attention
    A = torch.ones_like(graph)
    input_graph = A - graph
    index_list = input_graph.nonzero()

    edge_num = int(index_list.shape[0])
    edge_mask = (input_graph > 0).cpu()
    add_num = int(edge_num * add_percent)

    add_prob = sim_mx[edge_mask].cpu().numpy()
    add_prob /= add_prob.sum()
    drop_list = np.random.choice(edge_num, size=add_num, p=add_prob, replace=False)
    drop_index = index_list[drop_list]

    ones = torch.ones_like(aug_graph[0, 0])
    aug_graph[drop_index[:, 0], drop_index[:, 1]] = ones

    aug_graph = aug_graph + I
    return aug_graph

if __name__ == '__main__':
    N = 50
    T = 20
    D = 64
    input = torch.randn(50, 20, 64)
    st = ST_block(dim=64)
    ST_output = st(input)

    num_clusters = 3
    data = ST_output.detach().clone()
    data = data.view(N * T, D).numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(data)
    cluster_labels = torch.from_numpy(cluster_labels).view(N, T)
    cluster_centers = torch.from_numpy(kmeans.cluster_centers_)

    loss_reg = F.mse_loss(input, ST_output)

    ST_output_flat = ST_output.view(N * T, D)
    cluster_centers_flat = cluster_centers.view(num_clusters, D)
    cluster_center4feature = cluster_centers_flat.index_select(0, cluster_labels.view(-1))
    loss_global = torch.mean(torch.norm(ST_output_flat - cluster_center4feature, dim=1))

    loss = loss_reg + loss_global












