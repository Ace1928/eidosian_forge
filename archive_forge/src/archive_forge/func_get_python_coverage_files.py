from __future__ import annotations
import collections.abc as c
import json
import os
import re
import typing as t
from ...encoding import (
from ...io import (
from ...util import (
from ...util_common import (
from ...config import (
from ...python_requirements import (
from ...target import (
from ...data import (
from ...pypi_proxy import (
from ...provisioning import (
from ...coverage_util import (
def get_python_coverage_files(path: t.Optional[str]=None) -> list[str]:
    """Return the list of Python coverage file paths."""
    return get_coverage_files('python', path)