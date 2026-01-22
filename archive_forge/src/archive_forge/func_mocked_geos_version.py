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
@pytest.fixture
def mocked_geos_version():
    with mock.patch.object(shapely.lib, 'geos_version', new=(3, 7, 1)):
        yield '3.7.1'