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
def collect_units_install() -> list[PipInstall]:
    """Return details necessary for the specified units pip install(s)."""
    requirements_paths: list[tuple[str, str]] = []
    constraints_paths: list[tuple[str, str]] = []
    path = os.path.join(data_context().content.unit_path, 'requirements.txt')
    requirements_paths.append((data_context().content.root, path))
    path = os.path.join(data_context().content.unit_path, 'constraints.txt')
    constraints_paths.append((data_context().content.root, path))
    return collect_install(requirements_paths, constraints_paths)