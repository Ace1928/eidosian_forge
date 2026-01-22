import logging
from types import ModuleType
from typing import Any, Dict, List, Optional
from ray.autoscaler._private.command_runner import DockerCommandRunner, SSHCommandRunner
from ray.autoscaler.command_runner import CommandRunnerInterface
from ray.util.annotations import DeveloperAPI
def find_node_id():
    if use_internal_ip:
        return self._internal_ip_cache.get(ip_address)
    else:
        return self._external_ip_cache.get(ip_address)