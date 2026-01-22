import pkgutil
import types
import importlib
import warnings
from importlib import import_module
import pytest
import scipy
def is_unexpected(name):
    """Check if this needs to be considered."""
    if '._' in name or '.tests' in name or '.setup' in name:
        return False
    if name in PUBLIC_MODULES:
        return False
    if name in PRIVATE_BUT_PRESENT_MODULES:
        return False
    return True