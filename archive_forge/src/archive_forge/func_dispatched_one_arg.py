import inspect
import sys
import os
import tempfile
from io import StringIO
from unittest import mock
import numpy as np
from numpy.testing import (
from numpy.core.overrides import (
from numpy.compat import pickle
import pytest
@array_function_dispatch(lambda array: (array,))
def dispatched_one_arg(array):
    """Docstring."""
    return 'original'