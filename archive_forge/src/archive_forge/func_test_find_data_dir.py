import os
import sys
import tempfile
from os import environ as env
from os.path import join as pjoin
from tempfile import TemporaryDirectory
import pytest
from .. import data as nibd
from ..data import (
from .test_environment import DATA_KEY, USER_KEY, with_environment
def test_find_data_dir():
    here, fname = os.path.split(__file__)
    under_here, subhere = os.path.split(here)
    with pytest.raises(DataError):
        find_data_dir([here], 'implausible', 'directory')
    with pytest.raises(DataError):
        find_data_dir([here], fname)
    dd = find_data_dir([under_here], subhere)
    assert dd == here
    dud_dir = pjoin(under_here, 'implausible')
    dd = find_data_dir([dud_dir, under_here], subhere)
    assert dd == here