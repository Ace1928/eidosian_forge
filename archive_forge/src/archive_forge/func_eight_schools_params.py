import gzip
import importlib
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import cloudpickle
import numpy as np
import pytest
from _pytest.outcomes import Skipped
from packaging.version import Version
from ..data import InferenceData, from_dict
@pytest.fixture(scope='module')
def eight_schools_params():
    """Share setup for eight schools."""
    return {'J': 8, 'y': np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]), 'sigma': np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])}