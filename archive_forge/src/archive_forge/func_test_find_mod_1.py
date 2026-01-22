import shutil
import sys
import tempfile
from pathlib import Path
import IPython.utils.module_paths as mp
def test_find_mod_1():
    """
    Search for a directory's file path.
    Expected output: a path to that directory's __init__.py file.
    """
    modpath = TMP_TEST_DIR / 'xmod' / '__init__.py'
    assert Path(mp.find_mod('xmod')) == modpath