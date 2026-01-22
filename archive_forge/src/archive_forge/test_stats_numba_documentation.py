import importlib
import numpy as np
import pytest
from ...rcparams import rcParams
from ...stats import r2_score, summary
from ...utils import Numba
from ..helpers import (  # pylint: disable=unused-import
from .test_stats import centered_eight, non_centered_eight  # pylint: disable=unused-import
Numba test for r2_score