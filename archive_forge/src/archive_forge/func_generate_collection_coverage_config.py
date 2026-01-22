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
def generate_collection_coverage_config(args: TestConfig) -> str:
    """Generate code coverage configuration for Ansible Collection tests."""
    coverage_config = '\n[run]\nbranch = True\nconcurrency =\n    multiprocessing\n    thread\nparallel = True\ndisable_warnings =\n    no-data-collected\n'
    if isinstance(args, IntegrationConfig):
        coverage_config += '\ninclude =\n    %s/*\n    */%s/*\n' % (data_context().content.root, data_context().content.collection.directory)
    elif isinstance(args, SanityConfig):
        coverage_config += '\ninclude =\n    %s/*\n\nomit =\n    %s/*\n' % (data_context().content.root, os.path.join(data_context().content.root, data_context().content.results_path))
    else:
        coverage_config += '\ninclude =\n     %s/*\n' % data_context().content.root
    return coverage_config