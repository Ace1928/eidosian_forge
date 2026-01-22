from __future__ import annotations
import abc
import typing as t
from ...config import (
from ...util import (
from ...target import (
from ...host_configs import (
from ...host_profiles import (
def get_target_filter(args: IntegrationConfig, configs: list[HostConfig], controller: bool) -> TargetFilter:
    """Return an integration test target filter instance for the provided host configurations."""
    target_type = type(configs[0])
    if issubclass(target_type, ControllerConfig):
        target_type = type(args.controller)
        configs = [args.controller]
    filter_type = get_host_target_type_map()[target_type]
    filter_instance = filter_type(args, configs, controller)
    return filter_instance