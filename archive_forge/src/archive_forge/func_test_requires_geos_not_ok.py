import os
import sys
from inspect import cleandoc
from itertools import chain
from string import ascii_letters, digits
from unittest import mock
import numpy as np
import pytest
import shapely
from shapely.decorators import multithreading_enabled, requires_geos
@pytest.mark.parametrize('version', ['3.7.2', '3.8.0', '3.8.1'])
def test_requires_geos_not_ok(version, mocked_geos_version):
    wrapped = requires_geos(version)(func)
    with pytest.raises(shapely.errors.UnsupportedGEOSVersionError):
        wrapped()
    assert wrapped.__doc__ == expected_docstring(version=version, indent=' ' * 4)