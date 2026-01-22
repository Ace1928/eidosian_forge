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
def update_coverage_filename(original_filename: str, platform: str) -> str:
    """Validate the given filename and insert the specified platform, then return the result."""
    parts = original_filename.split('=')
    if original_filename != os.path.basename(original_filename) or len(parts) != 5 or parts[2] != 'platform':
        raise Exception(f'Unexpected coverage filename: {original_filename}')
    parts[2] = platform
    updated_filename = '='.join(parts)
    display.info(f'Coverage file for platform "{platform}": {original_filename} -> {updated_filename}', verbosity=3)
    return updated_filename