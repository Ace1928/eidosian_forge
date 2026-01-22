import io
from collections import defaultdict
from itertools import filterfalse
from typing import Dict, List, Tuple, Mapping, TypeVar
from .. import _reqs
from ..extern.jaraco.text import yield_lines
from ..extern.packaging.requirements import Requirement
def write_setup_requirements(cmd, basename, filename):
    data = io.StringIO()
    _write_requirements(data, cmd.distribution.setup_requires)
    cmd.write_or_delete_file('setup-requirements', filename, data.getvalue())