import os
import shutil
import subprocess
import sys
import pytest
import numpy as np
from numpy.testing import IS_WASM
def test_conv_intp(install_temp):
    import checks

    class myint:

        def __int__(self):
            return 3
    assert checks.conv_intp(3.0) == 3
    assert checks.conv_intp(myint()) == 3