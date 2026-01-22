import bz2
import gzip
import lzma
import os
import shutil
import struct
import tarfile
import warnings
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type, Union
from .. import config
from ._filelock import FileLock
from .logging import get_logger
@staticmethod
def read_magic_number(path: Union[Path, str], magic_number_length: int):
    with open(path, 'rb') as f:
        return f.read(magic_number_length)