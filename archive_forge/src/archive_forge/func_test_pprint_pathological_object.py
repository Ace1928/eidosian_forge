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
def test_pprint_pathological_object(self):
    """
        If the test fails, it at least won't hang.
        """

    class A:

        def __getitem__(self, key):
            return 3
    df = DataFrame([A()])
    repr(df)