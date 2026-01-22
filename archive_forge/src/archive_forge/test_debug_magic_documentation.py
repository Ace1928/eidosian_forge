import os
import sys
from IPython.testing.decorators import skip_win32
from IPython.testing import IPYTHON_TESTING_TIMEOUT_SCALE

    This test that we can correctly pass through frames of a generator post-mortem.
    