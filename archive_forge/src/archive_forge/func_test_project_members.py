from pyxnat import Interface
from requests.exceptions import ConnectionError
import os.path as op
from functools import wraps
import pytest
def test_project_members():
    x = Interface(config=fp)
    assert isinstance(x.select.project('pyxnat_tests').members(), list)