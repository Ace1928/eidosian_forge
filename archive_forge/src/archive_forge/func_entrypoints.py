from __future__ import annotations
import numbers
import sys
from contextlib import contextmanager
from functools import wraps
from importlib import metadata as importlib_metadata
from io import UnsupportedOperation
from kombu.exceptions import reraise
def entrypoints(namespace):
    """Return setuptools entrypoints for namespace."""
    if sys.version_info >= (3, 10):
        entry_points = importlib_metadata.entry_points(group=namespace)
    else:
        entry_points = importlib_metadata.entry_points()
        try:
            entry_points = entry_points.get(namespace, [])
        except AttributeError:
            entry_points = entry_points.select(group=namespace)
    return ((ep, ep.load()) for ep in entry_points)