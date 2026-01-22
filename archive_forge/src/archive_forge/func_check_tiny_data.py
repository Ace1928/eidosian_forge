import os
import io
import logging
import shutil
import stat
from pathlib import Path
from contextlib import contextmanager
from .. import __version__ as full_version
from ..utils import check_version, get_logger
def check_tiny_data(fname):
    """
    Load the tiny-data.txt file and check that the contents are correct.
    """
    assert os.path.exists(fname)
    with open(fname, encoding='utf-8') as tinydata:
        content = tinydata.read()
    true_content = '\n'.join(['# A tiny data file for test purposes only', '1  2  3  4  5  6'])
    assert content.strip() == true_content