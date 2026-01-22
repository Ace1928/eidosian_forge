from __future__ import annotations
import shutil
import typing as t
from .config import (
from .util import (
from .host_profiles import (
from .ssh import (
def create_network_inventory(args: EnvironmentConfig, path: str, target_hosts: list[HostProfile]) -> None:
    """Create and return inventory for use in target network integration tests."""
    first = target_hosts[0]
    if isinstance(first, NetworkInventoryProfile):
        if args.explain:
            return
        try:
            shutil.copyfile(first.config.path, path)
        except shutil.SameFileError:
            pass
        return
    target_hosts = t.cast(list[NetworkRemoteProfile], target_hosts)
    host_groups: dict[str, dict[str, dict[str, t.Union[str, int]]]] = {target_host.config.platform: {} for target_host in target_hosts}
    for target_host in target_hosts:
        host_groups[target_host.config.platform][sanitize_host_name(target_host.config.name)] = target_host.get_inventory_variables()
    inventory = Inventory(host_groups=host_groups, extra_groups={'net:children': sorted(host_groups)})
    inventory.write(args, path)