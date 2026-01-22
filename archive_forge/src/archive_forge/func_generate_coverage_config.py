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
def generate_coverage_config(args: TestConfig) -> str:
    """Generate code coverage configuration for tests."""
    if data_context().content.collection:
        coverage_config = generate_collection_coverage_config(args)
    else:
        coverage_config = generate_ansible_coverage_config()
    return coverage_config