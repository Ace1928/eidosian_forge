from abc import abstractmethod
from contextlib import closing
import functools
import hashlib
import multiprocessing
import multiprocessing.dummy
import os
import queue
import random
import shutil
import sys  # pylint: disable=unused-import
import tarfile
import threading
import time
import typing
import urllib
import weakref
import zipfile
import numpy as np
from tensorflow.python.framework import tensor
from six.moves.urllib.request import urlopen
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.keras.utils.io_utils import path_to_string
def chunk_read(response, chunk_size=8192, reporthook=None):
    content_type = response.info().get('Content-Length')
    total_size = -1
    if content_type is not None:
        total_size = int(content_type.strip())
    count = 0
    while True:
        chunk = response.read(chunk_size)
        count += 1
        if reporthook is not None:
            reporthook(count, chunk_size, total_size)
        if chunk:
            yield chunk
        else:
            break