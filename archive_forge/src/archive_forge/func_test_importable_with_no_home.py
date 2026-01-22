import os
import subprocess
import sys
import pytest
import matplotlib
def test_importable_with_no_home(tmpdir):
    subprocess.run([sys.executable, '-c', 'import pathlib; pathlib.Path.home = lambda *args: 1/0; import matplotlib.pyplot'], env={**os.environ, 'MPLCONFIGDIR': str(tmpdir)}, check=True)