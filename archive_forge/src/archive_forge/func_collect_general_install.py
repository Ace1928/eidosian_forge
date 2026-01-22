from __future__ import annotations
import base64
import dataclasses
import json
import os
import re
import typing as t
from .encoding import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .data import (
from .host_configs import (
from .connections import (
from .coverage_util import (
def collect_general_install(command: t.Optional[str]=None, ansible: bool=False) -> list[PipInstall]:
    """Return details necessary for the specified general-purpose pip install(s)."""
    requirements_paths: list[tuple[str, str]] = []
    constraints_paths: list[tuple[str, str]] = []
    if ansible:
        path = os.path.join(ANSIBLE_TEST_DATA_ROOT, 'requirements', 'ansible.txt')
        requirements_paths.append((ANSIBLE_TEST_DATA_ROOT, path))
    if command:
        path = os.path.join(ANSIBLE_TEST_DATA_ROOT, 'requirements', f'{command}.txt')
        requirements_paths.append((ANSIBLE_TEST_DATA_ROOT, path))
    return collect_install(requirements_paths, constraints_paths)