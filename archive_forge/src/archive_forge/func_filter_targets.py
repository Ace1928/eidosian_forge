from __future__ import annotations
import os
import typing as t
from . import (
from ...test import (
from ...target import (
from ...util import (
from ...util_common import (
from ...config import (
from ...host_configs import (
def filter_targets(self, targets: list[TestTarget]) -> list[TestTarget]:
    """Return the given list of test targets, filtered to include only those relevant for the test."""
    return [target for target in targets if os.path.splitext(target.path)[1] == '.py' or is_subdir(target.path, 'bin')]