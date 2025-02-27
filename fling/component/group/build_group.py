from fling.utils.registry_utils import GROUP_REGISTRY
from fling.utils import Logger
from fling.component.group import ParameterServerGroup


def get_group(args: dict, logger: Logger) -> ParameterServerGroup:
    return GROUP_REGISTRY.build(args.group.name, args, logger)
