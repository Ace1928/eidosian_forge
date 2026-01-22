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
def dl_progress(count, block_size, total_size):
    if ProgressTracker.progbar is None:
        if total_size == -1:
            total_size = None
        ProgressTracker.progbar = Progbar(total_size)
    else:
        ProgressTracker.progbar.update(count * block_size)