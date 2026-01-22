import shutil
import sys
import tempfile
from pathlib import Path
import IPython.utils.module_paths as mp
def test_find_mod_3():
    """
    Search for a directory + a filename without its .py extension
    Expected output: full path with .py extension.
    """
    modpath = TMP_TEST_DIR / 'xmod' / 'sub.py'
    assert Path(mp.find_mod('xmod.sub')) == modpath