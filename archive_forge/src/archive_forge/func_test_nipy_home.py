import os
from os import environ as env
from os.path import abspath
from os.path import join as pjoin
import pytest
from .. import environment as nibe
def test_nipy_home():
    assert nibe.get_home_dir() == os.path.expanduser('~')