import sys, platform, re, pytest
from numpy.core._multiarray_umath import (
import numpy as np
import subprocess
import pathlib
import os
import re
@pytest.mark.parametrize('enabled, disabled', [('feature', 'feature'), ('feature', 'same')])
def test_both_enable_disable_set(self, enabled, disabled):
    """
        Ensure that when both environment variables are set then an
        ImportError is thrown
        """
    self.env['NPY_ENABLE_CPU_FEATURES'] = enabled
    self.env['NPY_DISABLE_CPU_FEATURES'] = disabled
    msg = 'Both NPY_DISABLE_CPU_FEATURES and NPY_ENABLE_CPU_FEATURES'
    err_type = 'ImportError'
    self._expect_error(msg, err_type)