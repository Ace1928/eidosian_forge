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
def file_properties(self) -> Dict[str, Any]:
    """Return file properties as a hash object. Following keys are included: ‘readonly’, ‘is_directory’,
        ‘posix_mode’, ‘archivable’, ‘emptystream’, ‘filename’, ‘creationtime’, ‘lastaccesstime’,
        ‘lastwritetime’, ‘attributes’
        """
    properties = self._file_info
    if properties is not None:
        properties['readonly'] = self.readonly
        properties['posix_mode'] = self.posix_mode
        properties['archivable'] = self.archivable
        properties['is_directory'] = self.is_directory
    return properties