import time
import gzip
import struct
import traceback
import numbers
import sys
import os
import platform
import errno
import logging
import bz2
import zipfile
import json
from contextlib import contextmanager
from collections import OrderedDict
import numpy as np
import numpy.testing as npt
import numpy.random as rnd
import mxnet as mx
from .context import Context, current_context
from .ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID
from .ndarray import array
from .symbol import Symbol
from .symbol.numpy import _Symbol as np_symbol
from .util import use_np, getenv, setenv  # pylint: disable=unused-import
from .runtime import Features
from .numpy_extension import get_cuda_compute_capability
def get_zip_data(data_dir, url, data_origin_name):
    """Download and extract zip data.

    Parameters
    ----------

    data_dir : str
        Absolute or relative path of the directory name to store zip files
    url : str
        URL to download data from
    data_origin_name : str
        Name of the downloaded zip file

    Examples
    --------
    >>> get_zip_data("data_dir",
                     "http://files.grouplens.org/datasets/movielens/ml-10m.zip",
                     "ml-10m.zip")
    """
    data_origin_name = os.path.join(data_dir, data_origin_name)
    if not os.path.exists(data_origin_name):
        download(url, dirname=data_dir, overwrite=False)
        zip_file = zipfile.ZipFile(data_origin_name)
        zip_file.extractall(path=data_dir)