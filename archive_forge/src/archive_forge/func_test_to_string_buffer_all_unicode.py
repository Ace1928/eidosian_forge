from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
from pandas.io.formats import printing
import pandas.io.formats.format as fmt
def test_to_string_buffer_all_unicode(self):
    buf = StringIO()
    empty = DataFrame({'c/σ': Series(dtype=object)})
    nonempty = DataFrame({'c/σ': Series([1, 2, 3])})
    print(empty, file=buf)
    print(nonempty, file=buf)
    buf.getvalue()