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
def run_pypi_proxy(args: EnvironmentConfig, targets_use_pypi: bool) -> None:
    """Run a PyPI proxy support container."""
    if args.pypi_endpoint:
        return
    versions_needing_proxy: tuple[str, ...] = tuple()
    posix_targets = [target for target in args.targets if isinstance(target, PosixConfig)]
    need_proxy = targets_use_pypi and any((target.python.version in versions_needing_proxy for target in posix_targets))
    use_proxy = args.pypi_proxy or need_proxy
    if not use_proxy:
        return
    if not docker_available():
        if args.pypi_proxy:
            raise ApplicationError('Use of the PyPI proxy was requested, but Docker is not available.')
        display.warning('Unable to use the PyPI proxy because Docker is not available. Installation of packages using `pip` may fail.')
        return
    image = 'quay.io/ansible/pypi-test-container:2.0.0'
    port = 3141
    run_support_container(args=args, context='__pypi_proxy__', image=image, name='pypi-test-container', ports=[port])