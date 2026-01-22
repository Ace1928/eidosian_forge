import shutil
import sys
import tempfile
from pathlib import Path
import IPython.utils.module_paths as mp
def test_tempdir():
    """
    Ensure the test are done with a temporary file that have a dot somewhere.
    """
    assert '.' in str(TMP_TEST_DIR)