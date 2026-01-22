from __future__ import annotations
import os
import typing as t
import astroid
from pylint.checkers import BaseChecker
def is_module_path(path):
    """Return True if the given path is a module or module_utils path, otherwise return False."""
    return path.startswith(ANSIBLE_TEST_MODULES_PATH) or path.startswith(ANSIBLE_TEST_MODULE_UTILS_PATH)