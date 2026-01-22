import fnmatch
import os
import os.path
import random
import sys
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from typing import Any, BinaryIO, Generator, List, Union, cast
from pip._vendor.tenacity import retry, stop_after_delay, wait_fixed
from pip._internal.utils.compat import get_path_uid
from pip._internal.utils.misc import format_size
def directory_size(path: str) -> Union[int, float]:
    size = 0.0
    for root, _dirs, files in os.walk(path):
        for filename in files:
            file_path = os.path.join(root, filename)
            size += file_size(file_path)
    return size