import os
import subprocess
import sys
from IPython.utils.tempdir import NamedFileInTemporaryDirectory
from IPython.testing.decorators import skip_win32
from IPython.testing import IPYTHON_TESTING_TIMEOUT_SCALE
import IPython
test that `IPython.embed()` is nestable