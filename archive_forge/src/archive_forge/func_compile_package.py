import argparse
import functools
import itertools
import marshal
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List
@indent_msg
def compile_package(self, path: Path, top_package_path: Path):
    """Compile all the files within a Python package dir."""
    assert path.is_dir()
    if path.name in DENY_LIST:
        self.msg(path, 'X')
        return
    is_package_dir = any((child.name == '__init__.py' for child in path.iterdir()))
    if not is_package_dir:
        self.msg(path, 'S')
        return
    self.msg(path, 'P')
    for child in path.iterdir():
        self.compile_path(child, top_package_path)