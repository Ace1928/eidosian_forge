import shlex
import subprocess
import time
import uuid
import pytest
from pandas.compat import (
import pandas.util._test_decorators as td
import pandas.io.common as icom
from pandas.io.parsers import read_csv
@pytest.fixture
def jsonl_file(datapath):
    """Path to a JSONL dataset"""
    return datapath('io', 'parser', 'data', 'items.jsonl')