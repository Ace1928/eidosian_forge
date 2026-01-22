import builtins
import sys
import types
from unittest import SkipTest, mock
import pytest
from packaging.version import Version
from nibabel.optpkg import optional_package
from nibabel.tripwire import TripWire, TripWireError
Testing optpkg module
