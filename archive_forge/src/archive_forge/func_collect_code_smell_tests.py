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
@cache
def collect_code_smell_tests() -> tuple[SanityTest, ...]:
    """Return a tuple of available code smell sanity tests."""
    paths = glob.glob(os.path.join(SANITY_ROOT, 'code-smell', '*.py'))
    if data_context().content.is_ansible:
        ansible_code_smell_root = os.path.join(data_context().content.root, 'test', 'sanity', 'code-smell')
        skip_tests = read_lines_without_comments(os.path.join(ansible_code_smell_root, 'skip.txt'), remove_blank_lines=True, optional=True)
        paths.extend((path for path in glob.glob(os.path.join(ansible_code_smell_root, '*.py')) if os.path.basename(path) not in skip_tests))
    tests = tuple((SanityCodeSmellTest(p) for p in paths))
    return tests