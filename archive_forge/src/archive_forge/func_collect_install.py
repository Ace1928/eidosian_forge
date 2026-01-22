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
def collect_install(requirements_paths: list[tuple[str, str]], constraints_paths: list[tuple[str, str]], packages: t.Optional[list[str]]=None, constraints: bool=True) -> list[PipInstall]:
    """Build a pip install list from the given requirements, constraints and packages."""
    constraints_paths = list(constraints_paths)
    if constraints:
        constraints_paths.append((ANSIBLE_TEST_DATA_ROOT, os.path.join(ANSIBLE_TEST_DATA_ROOT, 'requirements', 'constraints.txt')))
    requirements = [(os.path.relpath(path, root), read_text_file(path)) for root, path in requirements_paths if usable_pip_file(path)]
    constraints = [(os.path.relpath(path, root), read_text_file(path)) for root, path in constraints_paths if usable_pip_file(path)]
    packages = packages or []
    if requirements or packages:
        installs = [PipInstall(requirements=requirements, constraints=constraints, packages=packages)]
    else:
        installs = []
    return installs