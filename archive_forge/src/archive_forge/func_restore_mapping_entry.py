import contextlib
import os
import shutil
import sys
import tempfile
import time
import unittest
from traits.etsconfig.etsconfig import ETSConfig, ETSToolkitError
@contextlib.contextmanager
def restore_mapping_entry(mapping, key):
    """
    Context manager that restores a mapping entry to its previous
    state on exit.

    """
    missing = object()
    old_value = mapping.get(key, missing)
    try:
        yield
    finally:
        if old_value is missing:
            mapping.pop(key, None)
        else:
            mapping[key] = old_value