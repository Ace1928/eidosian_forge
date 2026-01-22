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
@pytest.mark.skipif(sys.platform.startswith('win') and (shapely.geos_version == (3, 6, 6) or shapely.geos_version[:2] == (3, 7)), reason='GEOS_C_API_VERSION broken for GEOS 3.6.6 and 3.7.x on Windows')
def test_geos_capi_version():
    expected = '{}.{}.{}-CAPI-{}.{}.{}'.format(*shapely.geos_version + shapely.geos_capi_version)
    actual_geos_version, actual_geos_api_version = shapely.geos_capi_version_string.split('-CAPI-')
    if any((c.isalpha() for c in actual_geos_version)):
        if actual_geos_version[-1].isnumeric():
            actual_geos_version = actual_geos_version.rstrip(digits)
        actual_geos_version = actual_geos_version.rstrip(ascii_letters)
    actual_geos_version = actual_geos_version.rstrip(ascii_letters)
    assert f'{actual_geos_version}-CAPI-{actual_geos_api_version}' == expected