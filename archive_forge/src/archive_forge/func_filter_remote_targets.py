from __future__ import annotations
import abc
import glob
import hashlib
import json
import os
import pathlib
import re
import collections
import collections.abc as c
import typing as t
from ...constants import (
from ...encoding import (
from ...io import (
from ...util import (
from ...util_common import (
from ...ansible_util import (
from ...target import (
from ...executor import (
from ...python_requirements import (
from ...config import (
from ...test import (
from ...data import (
from ...content_config import (
from ...host_configs import (
from ...host_profiles import (
from ...provisioning import (
from ...pypi_proxy import (
from ...venv import (
@staticmethod
def filter_remote_targets(targets: list[TestTarget]) -> list[TestTarget]:
    """Return a filtered list of the given targets, including only those that require support for remote-only Python versions."""
    targets = [target for target in targets if is_subdir(target.path, data_context().content.module_path) or is_subdir(target.path, data_context().content.module_utils_path) or is_subdir(target.path, data_context().content.unit_module_path) or is_subdir(target.path, data_context().content.unit_module_utils_path) or re.search('^%s/.*/library/' % re.escape(data_context().content.integration_targets_path), target.path) or (data_context().content.is_ansible and (is_subdir(target.path, 'test/lib/ansible_test/_util/target/') or re.search('^test/support/integration/.*/(modules|module_utils)/', target.path) or re.search('^lib/ansible/utils/collection_loader/', target.path)))]
    return targets