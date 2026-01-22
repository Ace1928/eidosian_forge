from __future__ import annotations
import inspect
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from .base import TestBase
from .. import config
from ..assertions import eq_
from ... import util
@staticmethod
def file_combinations(dirname):
    if os.path.isabs(dirname):
        path = dirname
    else:
        caller_path = inspect.stack()[1].filename
        path = os.path.join(os.path.dirname(caller_path), dirname)
    files = list(Path(path).glob('**/*.py'))
    for extra_dir in config.options.mypy_extra_test_paths:
        if extra_dir and os.path.isdir(extra_dir):
            files.extend((Path(extra_dir) / dirname).glob('**/*.py'))
    return files