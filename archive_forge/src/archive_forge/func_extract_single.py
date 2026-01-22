import collections.abc
import contextlib
import datetime
import errno
import functools
import io
import os
import pathlib
import queue
import re
import stat
import sys
import time
from multiprocessing import Process
from threading import Thread
from typing import IO, Any, BinaryIO, Collection, Dict, List, Optional, Tuple, Type, Union
import multivolumefile
from py7zr.archiveinfo import Folder, Header, SignatureHeader
from py7zr.callbacks import ExtractCallback
from py7zr.compressor import SupportedMethods, get_methods_names
from py7zr.exceptions import (
from py7zr.helpers import (
from py7zr.properties import DEFAULT_FILTERS, FILTER_DEFLATE64, MAGIC_7Z, get_default_blocksize, get_memory_limit
def extract_single(self, fp: Union[BinaryIO, str], files, path, src_start: int, src_end: int, q: Optional[queue.Queue], exc_q: Optional[queue.Queue]=None, skip_notarget=True) -> None:
    """
        Single thread extractor that takes file lists in single 7zip folder.
        """
    if files is None:
        return
    try:
        if isinstance(fp, str):
            fp = open(fp, 'rb')
        fp.seek(src_start)
        self._extract_single(fp, files, path, src_end, q, skip_notarget)
    except Exception as e:
        if exc_q is None:
            raise e
        else:
            exc_tuple = sys.exc_info()
            exc_q.put(exc_tuple)