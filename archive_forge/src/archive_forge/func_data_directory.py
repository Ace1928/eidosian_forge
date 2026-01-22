import os
import sys
import tempfile
from glob import glob
import numpy as np
import pytest
from ... import from_cmdstanpy
from ..helpers import (  # pylint: disable=unused-import
@pytest.fixture(scope='session')
def data_directory(self):
    here = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(here, '..', 'saved_models')
    return data_directory