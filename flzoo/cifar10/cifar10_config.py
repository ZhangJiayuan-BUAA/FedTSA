from easydict import EasyDict

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
exp_args = dict(
    data=dict(dataset='cifar10_test',
              data_path='./flzoo/cifar10/data/CIFAR10', # Original dataset, please replace yourself
              sample_method=dict(name='iid', train_num=50000, test_num=500),
              corruption=common_corruptions,
              partition_path='./flzoo/4area.npy',       # Data partitioning mode
              level=[5],
              class_number=10),

    learn=dict(
        device='cuda:0', batch_size=64,
    ),

    model=dict(
        name='resnet8',
        input_channel=3,
        class_number=10,
    ),

    client=dict(name='fedtta_client',
                client_num=20),

    server=dict(name='base_server'),

    group=dict(name='adapt_group',
               aggregation_method='st'),

    other=dict(logging_path='./flzoo/cifar10/logging/cifar10_fedtsa',
               model_path='./flzoo/cifar10/pretrain/resnet8_cifar10.ckpt',  # A pre-trained model on the cifar10 dataset
               online=True,
               adap_iter=1,
               ttt_batch=10,
               is_continue=True,
               niid=True,
               is_average=True,

               method='adapt',
               pre_trained='resnet8',
               resume=True,

               time_slide=5,
               st_lr=1e-4,
               st_epoch=100,
               robust_weight=0.1,
               st_head=1,
               loop=1,
               ),

)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import FedTTA_Pipeline
    FedTTA_Pipeline(exp_args, seed=0)
