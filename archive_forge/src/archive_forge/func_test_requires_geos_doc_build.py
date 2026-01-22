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
@pytest.mark.parametrize('version', ['3.6.0', '3.8.0'])
def test_requires_geos_doc_build(version, mocked_geos_version, sphinx_doc_build):
    """The requires_geos decorator always adapts the docstring."""
    wrapped = requires_geos(version)(func)
    assert wrapped.__doc__ == expected_docstring(version=version, indent=' ' * 4)