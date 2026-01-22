import logging
import socket
from functools import lru_cache
from typing import Optional
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_fabric.plugins.environments.lightning import find_free_network_port
@property
@override
def main_port(self) -> int:
    if self._main_port is None:
        self._main_port = self._get_main_port()
    return self._main_port