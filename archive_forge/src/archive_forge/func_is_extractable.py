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
@classmethod
def is_extractable(cls, path: Union[Path, str], return_extractor: bool=False) -> bool:
    warnings.warn("Method 'is_extractable' was deprecated in version 2.4.0 and will be removed in 3.0.0. Use 'infer_extractor_format' instead.", category=FutureWarning)
    extractor_format = cls.infer_extractor_format(path)
    if extractor_format:
        return True if not return_extractor else (True, cls.extractors[extractor_format])
    return False if not return_extractor else (False, None)