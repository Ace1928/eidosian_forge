import sys, platform, re, pytest
from numpy.core._multiarray_umath import (
import numpy as np
import subprocess
import pathlib
import os
import re
@pytest.mark.skipif(not __cpu_dispatch__, reason='NPY_*_CPU_FEATURES only parsed if `__cpu_dispatch__` is non-empty')
@pytest.mark.parametrize('action', ['ENABLE', 'DISABLE'])
def test_variable_too_long(self, action):
    """
        Test that an error is thrown if the environment variables are too long
        to be processed. Current limit is 1024, but this may change later.
        """
    MAX_VAR_LENGTH = 1024
    self.env[f'NPY_{action}_CPU_FEATURES'] = 't' * MAX_VAR_LENGTH
    msg = f"Length of environment variable 'NPY_{action}_CPU_FEATURES' is {MAX_VAR_LENGTH + 1}, only {MAX_VAR_LENGTH} accepted"
    err_type = 'RuntimeError'
    self._expect_error(msg, err_type)