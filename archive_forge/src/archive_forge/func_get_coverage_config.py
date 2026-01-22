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
@mutex
def get_coverage_config(args: TestConfig) -> str:
    """Return the path to the coverage config, creating the config if it does not already exist."""
    try:
        return get_coverage_config.path
    except AttributeError:
        pass
    coverage_config = generate_coverage_config(args)
    if args.explain:
        temp_dir = '/tmp/coverage-temp-dir'
    else:
        temp_dir = tempfile.mkdtemp()
        ExitHandler.register(lambda: remove_tree(temp_dir))
    path = os.path.join(temp_dir, COVERAGE_CONFIG_NAME)
    if not args.explain:
        write_text_file(path, coverage_config)
    get_coverage_config.path = path
    return path