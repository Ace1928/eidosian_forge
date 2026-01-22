from io import (
import os
import tempfile
import uuid
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_utf16_example(all_parsers, csv_dir_path):
    path = os.path.join(csv_dir_path, 'utf16_ex.txt')
    parser = all_parsers
    result = parser.read_csv(path, encoding='utf-16', sep='\t')
    assert len(result) == 50