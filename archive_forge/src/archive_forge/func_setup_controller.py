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
def setup_controller(self) -> None:
    """Perform setup for code coverage on the controller."""
    coverage_config_path = os.path.join(self.common_temp_path, COVERAGE_CONFIG_NAME)
    coverage_output_path = os.path.join(self.common_temp_path, ResultType.COVERAGE.name)
    coverage_config = generate_coverage_config(self.args)
    write_text_file(coverage_config_path, coverage_config, create_directories=True)
    verified_chmod(coverage_config_path, MODE_FILE)
    os.mkdir(coverage_output_path)
    verified_chmod(coverage_output_path, MODE_DIRECTORY_WRITE)