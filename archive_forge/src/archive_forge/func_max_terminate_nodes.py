import logging
from types import ModuleType
from typing import Any, Dict, List, Optional
from ray.autoscaler._private.command_runner import DockerCommandRunner, SSHCommandRunner
from ray.autoscaler.command_runner import CommandRunnerInterface
from ray.util.annotations import DeveloperAPI
@property
def max_terminate_nodes(self) -> Optional[int]:
    """The maximum number of nodes which can be terminated in one single
        API request. By default, this is "None", which means that the node
        provider's underlying API allows infinite requests to be terminated
        with one request.

        For example, AWS only allows 1000 nodes to be terminated
        at once; to terminate more, we must issue multiple separate API
        requests. If the limit is infinity, then simply set this to None.

        This may be overridden. The value may be useful when overriding the
        "terminate_nodes" method.
        """
    return None