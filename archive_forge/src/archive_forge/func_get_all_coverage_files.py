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
def get_all_coverage_files() -> list[str]:
    """Return a list of all coverage file paths."""
    return get_python_coverage_files() + get_powershell_coverage_files()