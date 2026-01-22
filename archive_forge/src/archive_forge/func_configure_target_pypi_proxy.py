from __future__ import annotations
import os
import urllib.parse
from .io import (
from .config import (
from .host_configs import (
from .util import (
from .util_common import (
from .docker_util import (
from .containers import (
from .ansible_util import (
from .host_profiles import (
from .inventory import (
def configure_target_pypi_proxy(args: EnvironmentConfig, profile: HostProfile, pypi_endpoint: str, pypi_hostname: str) -> None:
    """Configure the target environment to use a PyPI proxy."""
    inventory_path = process_scoped_temporary_file(args)
    create_posix_inventory(args, inventory_path, [profile])

    def cleanup_pypi_proxy() -> None:
        """Undo changes made to configure the PyPI proxy."""
        run_playbook(args, inventory_path, 'pypi_proxy_restore.yml', capture=True)
    force = 'yes' if profile.config.is_managed else 'no'
    run_playbook(args, inventory_path, 'pypi_proxy_prepare.yml', capture=True, variables=dict(pypi_endpoint=pypi_endpoint, pypi_hostname=pypi_hostname, force=force))
    ExitHandler.register(cleanup_pypi_proxy)