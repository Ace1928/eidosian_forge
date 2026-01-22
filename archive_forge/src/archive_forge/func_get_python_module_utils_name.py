from __future__ import annotations
import ast
import os
import re
import typing as t
from ..io import (
from ..util import (
from ..data import (
from ..target import (
def get_python_module_utils_name(path: str) -> str:
    """Return a namespace and name from the given module_utils path."""
    base_path = data_context().content.module_utils_path
    if data_context().content.collection:
        prefix = 'ansible_collections.' + data_context().content.collection.prefix + 'plugins.module_utils'
    else:
        prefix = 'ansible.module_utils'
    if path.endswith('/__init__.py'):
        path = os.path.dirname(path)
    if path == base_path:
        name = prefix
    else:
        name = prefix + '.' + os.path.splitext(os.path.relpath(path, base_path))[0].replace(os.path.sep, '.')
    return name