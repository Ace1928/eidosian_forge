import contextlib
from importlib import import_module
import os
import sys
from . import _util
@contextlib.contextmanager
def vendored(project, root=None):
    """A context manager under which the vendored project will be imported."""
    if root is None:
        root = project_root(project)
    sys.path.insert(0, root)
    try:
        yield root
    finally:
        sys.path.remove(root)