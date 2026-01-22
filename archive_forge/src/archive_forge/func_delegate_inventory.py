from __future__ import annotations
import collections.abc as c
import contextlib
import datetime
import json
import os
import re
import shutil
import tempfile
import time
import typing as t
from ...encoding import (
from ...ansible_util import (
from ...executor import (
from ...python_requirements import (
from ...ci import (
from ...target import (
from ...config import (
from ...io import (
from ...util import (
from ...util_common import (
from ...coverage_util import (
from ...cache import (
from .cloud import (
from ...data import (
from ...host_configs import (
from ...host_profiles import (
from ...provisioning import (
from ...pypi_proxy import (
from ...inventory import (
from .filters import (
from .coverage import (
def delegate_inventory(args: IntegrationConfig, inventory_path_src: str) -> None:
    """Make the given inventory available during delegation."""
    if isinstance(args, PosixIntegrationConfig):
        return

    def inventory_callback(payload_config: PayloadConfig) -> None:
        """
        Add the inventory file to the payload file list.
        This will preserve the file during delegation even if it is ignored or is outside the content and install roots.
        """
        files = payload_config.files
        inventory_path = get_inventory_relative_path(args)
        inventory_tuple = (inventory_path_src, inventory_path)
        if os.path.isfile(inventory_path_src) and inventory_tuple not in files:
            originals = [item for item in files if item[1] == inventory_path]
            if originals:
                for original in originals:
                    files.remove(original)
                display.warning('Overriding inventory file "%s" with "%s".' % (inventory_path, inventory_path_src))
            else:
                display.notice('Sourcing inventory file "%s" from "%s".' % (inventory_path, inventory_path_src))
            files.append(inventory_tuple)
    data_context().register_payload_callback(inventory_callback)