import gzip
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import textwrap
import time
import zipfile
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
import pandas._testing as tm
import pandas.io.common as icom
@pytest.mark.single_cpu
def test_with_missing_lzma():
    """Tests if import pandas works when lzma is not present."""
    code = textwrap.dedent("        import sys\n        sys.modules['lzma'] = None\n        import pandas\n        ")
    subprocess.check_output([sys.executable, '-c', code], stderr=subprocess.PIPE)