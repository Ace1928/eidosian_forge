import subprocess
import pytest
import sys
import os
import textwrap
import rpy2.rinterface as ri
def test_instantiate_without_initr():
    pycode = _instantiate_without_initr if os.name == 'nt' else _instantiate_without_initr.encode('ASCII')
    output = subprocess.check_output((sys.executable, '-c', pycode))
    assert output.rstrip() == b'Error: R not ready.'.rstrip()