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
def read_python_coverage_legacy(path: str) -> PythonArcs:
    """Return coverage arcs from the specified coverage file, which must be in the legacy JSON format."""
    try:
        contents = read_text_file(path)
        contents = re.sub("^!coverage.py: This is a private format, don't read it directly!", '', contents)
        data = json.loads(contents)
        arcs: PythonArcs = {filename: [t.cast(tuple[int, int], tuple(arc)) for arc in arc_list] for filename, arc_list in data['arcs'].items()}
    except Exception as ex:
        raise CoverageError(path, f'Error reading JSON coverage file: {ex}') from ex
    return arcs