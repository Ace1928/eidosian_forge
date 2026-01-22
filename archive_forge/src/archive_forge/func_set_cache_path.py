import hashlib
import logging
import numpy as np
import os
import pandas as pd
import tarfile
import tempfile
import six
import shutil
from .core import PATH_TYPES, fspath
def set_cache_path(path):
    assert isinstance(path, PATH_TYPES), 'expected string or pathlib.Path'
    global _cache_path
    _cache_path = fspath(path)