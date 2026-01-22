from __future__ import annotations
import collections.abc as c
import itertools
import json
import os
import datetime
import configparser
import typing as t
from . import (
from ...constants import (
from ...io import (
from ...test import (
from ...target import (
from ...util import (
from ...util_common import (
from ...ansible_util import (
from ...config import (
from ...data import (
from ...host_configs import (
def create_min_python_db(self, args: SanityConfig, targets: t.Iterable[TestTarget]) -> str:
    """Create a database of target file paths and their minimum required Python version, returning the path to the database."""
    target_paths = set((target.path for target in self.filter_remote_targets(list(targets))))
    controller_min_version = CONTROLLER_PYTHON_VERSIONS[0]
    target_min_version = REMOTE_ONLY_PYTHON_VERSIONS[0]
    min_python_versions = {os.path.abspath(target.path): target_min_version if target.path in target_paths else controller_min_version for target in targets}
    min_python_version_db_path = process_scoped_temporary_file(args)
    with open(min_python_version_db_path, 'w') as database_file:
        json.dump(min_python_versions, database_file)
    return min_python_version_db_path