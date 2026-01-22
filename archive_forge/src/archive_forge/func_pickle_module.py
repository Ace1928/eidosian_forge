import functools
import os
import pathlib
import subprocess
import sys
import time
import urllib.request
import pytest
import hypothesis as h
from ..conftest import groups, defaults
from pyarrow import set_timezone_db_path
from pyarrow.util import find_free_port
@pytest.fixture(params=['builtin_pickle', 'cloudpickle'], scope='session')
def pickle_module(request):
    return request.getfixturevalue(request.param)