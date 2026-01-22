import inspect
import re
import warnings
import pytest
from numpy.testing import assert_equal
from skimage._shared.testing import (
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from warnings import warn
@run_in_parallel(warnings_matching=['Test warning for test parallel'])
def change_state_warns_passes():
    warn('Test warning for test parallel', stacklevel=2)