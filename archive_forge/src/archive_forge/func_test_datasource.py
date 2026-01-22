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
def test_datasource():
    pth = pjoin('some', 'path')
    ds = Datasource(pth)
    assert ds.get_filename('unlikeley') == pjoin(pth, 'unlikeley')
    assert ds.get_filename('un', 'like', 'ley') == pjoin(pth, 'un', 'like', 'ley')