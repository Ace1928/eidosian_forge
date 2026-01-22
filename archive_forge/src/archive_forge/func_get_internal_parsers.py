from __future__ import annotations
import typing as t
from ...constants import (
from ...ci import (
from ...host_configs import (
from ..argparsing.parsers import (
from .value_parsers import (
from .host_config_parsers import (
from .base_argument_parsers import (
def get_internal_parsers(self, targets: list[NetworkConfig]) -> dict[str, Parser]:
    """Return a dictionary of type names and type parsers."""
    parsers: dict[str, Parser] = {}
    if self.allow_inventory and (not targets):
        parsers.update(inventory=NetworkInventoryParser())
    if not targets or not any((isinstance(target, NetworkInventoryConfig) for target in targets)):
        if get_ci_provider().supports_core_ci_auth():
            parsers.update(remote=NetworkRemoteParser())
    return parsers