import os
import subprocess
import sys
import warnings
import numpy as np
import unittest
from numba import jit
from numba.core.errors import (
from numba.core import errors
from numba.tests.support import ignore_internal_warnings
def return_external_array():
    return y