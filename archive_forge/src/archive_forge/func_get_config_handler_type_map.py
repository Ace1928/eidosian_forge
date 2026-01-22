from __future__ import annotations
import abc
import os
import shutil
import tempfile
import typing as t
import zipfile
from ...io import (
from ...ansible_util import (
from ...config import (
from ...util import (
from ...util_common import (
from ...coverage_util import (
from ...host_configs import (
from ...data import (
from ...host_profiles import (
from ...provisioning import (
from ...connections import (
from ...inventory import (
@cache
def get_config_handler_type_map() -> dict[t.Type[HostConfig], t.Type[CoverageHandler]]:
    """Create and return a mapping of HostConfig types to CoverageHandler types."""
    return get_type_map(CoverageHandler, HostConfig)