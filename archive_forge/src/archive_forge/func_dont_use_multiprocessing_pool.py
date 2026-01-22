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
def dont_use_multiprocessing_pool(f):

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        with _FORCE_THREADPOOL_LOCK:
            global _FORCE_THREADPOOL
            old_force_threadpool, _FORCE_THREADPOOL = (_FORCE_THREADPOOL, True)
            out = f(*args, **kwargs)
            _FORCE_THREADPOOL = old_force_threadpool
            return out
    return wrapped