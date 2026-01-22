from __future__ import annotations
import shutil
import typing as t
from .config import (
from .util import (
from .host_profiles import (
from .ssh import (
def create_windows_inventory(args: EnvironmentConfig, path: str, target_hosts: list[HostProfile]) -> None:
    """Create and return inventory for use in target Windows integration tests."""
    first = target_hosts[0]
    if isinstance(first, WindowsInventoryProfile):
        if args.explain:
            return
        try:
            shutil.copyfile(first.config.path, path)
        except shutil.SameFileError:
            pass
        return
    target_hosts = t.cast(list[WindowsRemoteProfile], target_hosts)
    hosts = [(target_host, target_host.wait_for_instance().connection) for target_host in target_hosts]
    windows_hosts = {sanitize_host_name(host.config.name): host.get_inventory_variables() for host, connection in hosts}
    inventory = Inventory(host_groups=dict(windows=windows_hosts), extra_groups={'testhost:children': ['windows']})
    inventory.write(args, path)