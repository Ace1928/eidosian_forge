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
def configure_pypi_proxy_easy_install(args: EnvironmentConfig, profile: HostProfile, pypi_endpoint: str) -> None:
    """Configure a custom index for easy_install based installs."""
    pydistutils_cfg_path = os.path.expanduser('~/.pydistutils.cfg')
    pydistutils_cfg = '\n[easy_install]\nindex_url = {0}\n'.format(pypi_endpoint).strip()
    if os.path.exists(pydistutils_cfg_path) and (not profile.config.is_managed):
        raise ApplicationError('Refusing to overwrite existing file: %s' % pydistutils_cfg_path)

    def pydistutils_cfg_cleanup() -> None:
        """Remove custom PyPI config."""
        display.info('Removing custom PyPI config: %s' % pydistutils_cfg_path, verbosity=1)
        os.remove(pydistutils_cfg_path)
    display.info('Injecting custom PyPI config: %s' % pydistutils_cfg_path, verbosity=1)
    display.info('Config: %s\n%s' % (pydistutils_cfg_path, pydistutils_cfg), verbosity=3)
    if not args.explain:
        write_text_file(pydistutils_cfg_path, pydistutils_cfg, True)
        ExitHandler.register(pydistutils_cfg_cleanup)