from __future__ import annotations
import shutil
import typing as t
from .config import (
from .util import (
from .host_profiles import (
from .ssh import (
def create_controller_inventory(args: EnvironmentConfig, path: str, controller_host: ControllerHostProfile) -> None:
    """Create and return inventory for use in controller-only integration tests."""
    inventory = Inventory(host_groups=dict(testgroup=dict(testhost=dict(ansible_connection='local', ansible_pipelining='yes', ansible_python_interpreter=controller_host.python.path))))
    inventory.write(args, path)