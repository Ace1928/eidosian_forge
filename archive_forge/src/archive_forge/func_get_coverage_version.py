from __future__ import annotations
import dataclasses
import os
import sqlite3
import tempfile
import typing as t
from .config import (
from .io import (
from .util import (
from .data import (
from .util_common import (
from .host_configs import (
from .constants import (
from .thread import (
def get_coverage_version(version: str) -> CoverageVersion:
    """Return the coverage version to use with the specified Python version."""
    python_version = str_to_version(version)
    supported_versions = [entry for entry in COVERAGE_VERSIONS if entry.min_python <= python_version <= entry.max_python]
    if not supported_versions:
        raise InternalError(f'Python {version} has no matching entry in COVERAGE_VERSIONS.')
    if len(supported_versions) > 1:
        raise InternalError(f'Python {version} has multiple matching entries in COVERAGE_VERSIONS.')
    coverage_version = supported_versions[0]
    return coverage_version