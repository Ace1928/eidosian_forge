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
@pytest.mark.parametrize('version', ['3.7.0', '3.7.1', '3.6.2'])
def test_requires_geos_ok(version, mocked_geos_version):
    wrapped = requires_geos(version)(func)
    assert wrapped is func