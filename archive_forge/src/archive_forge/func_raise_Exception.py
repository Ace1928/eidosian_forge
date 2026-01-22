import builtins
import sys
import types
from unittest import SkipTest, mock
import pytest
from packaging.version import Version
from nibabel.optpkg import optional_package
from nibabel.tripwire import TripWire, TripWireError
def raise_Exception(*args, **kwargs):
    if args[0] == 'nottriedbefore':
        raise Exception('non ImportError could be thrown by some malfunctioning module upon import, and optional_package should catch it too')
    return orig_import(*args, **kwargs)