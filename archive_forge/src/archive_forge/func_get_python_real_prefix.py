from __future__ import annotations
import collections.abc as c
import json
import os
import pathlib
import sys
import typing as t
from .config import (
from .util import (
from .util_common import (
from .host_configs import (
from .python_requirements import (
def get_python_real_prefix(python_path: str) -> t.Optional[str]:
    """
    Return the real prefix of the specified interpreter or None if the interpreter is not a virtual environment created by 'virtualenv'.
    """
    cmd = [python_path, os.path.join(os.path.join(ANSIBLE_TEST_TARGET_TOOLS_ROOT, 'virtualenvcheck.py'))]
    check_result = json.loads(raw_command(cmd, capture=True)[0])
    real_prefix = check_result['real_prefix']
    return real_prefix