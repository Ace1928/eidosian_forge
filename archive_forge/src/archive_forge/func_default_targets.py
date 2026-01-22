from __future__ import annotations
import argparse
import collections.abc as c
import dataclasses
import enum
import os
import types
import typing as t
from ..constants import (
from ..util import (
from ..docker_util import (
from ..completion import (
from ..host_configs import (
from ..data import (
def default_targets(mode: TargetMode, controller: ControllerHostConfig) -> list[HostConfig]:
    """Return a list of default targets for the given target mode."""
    targets: list[HostConfig]
    if mode == TargetMode.WINDOWS_INTEGRATION:
        targets = [WindowsInventoryConfig(path=os.path.abspath(os.path.join(data_context().content.integration_path, 'inventory.winrm')))]
    elif mode == TargetMode.NETWORK_INTEGRATION:
        targets = [NetworkInventoryConfig(path=os.path.abspath(os.path.join(data_context().content.integration_path, 'inventory.networking')))]
    elif mode.multiple_pythons:
        targets = t.cast(list[HostConfig], controller.get_default_targets(HostContext(controller_config=controller)))
    else:
        targets = [ControllerConfig()]
    return targets