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
@pytest.mark.xfail(raises=DataError)
def test_bomber():
    b = Bomber('bomber example', 'a message')
    b.any_attribute