import atexit
import functools
import hashlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory, TemporaryFile
import weakref
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import cbook
from matplotlib.testing.exceptions import ImageComparisonFailure
def get_file_hash(path, block_size=2 ** 20):
    md5 = hashlib.md5()
    with open(path, 'rb') as fd:
        while True:
            data = fd.read(block_size)
            if not data:
                break
            md5.update(data)
    if Path(path).suffix == '.pdf':
        md5.update(str(mpl._get_executable_info('gs').version).encode('utf-8'))
    elif Path(path).suffix == '.svg':
        md5.update(str(mpl._get_executable_info('inkscape').version).encode('utf-8'))
    return md5.hexdigest()