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
@array_function_dispatch(lambda array, option=None: (array,))
def func_with_option(array, option='default'):
    return option