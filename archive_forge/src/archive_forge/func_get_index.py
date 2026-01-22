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
def get_index(uid, i):
    """Get the value from the Sequence `uid` at index `i`.

  To allow multiple Sequences to be used at the same time, we use `uid` to
  get a specific one. A single Sequence would cause the validation to
  overwrite the training Sequence.

  Args:
      uid: int, Sequence identifier
      i: index

  Returns:
      The value at index `i`.
  """
    return _SHARED_SEQUENCES[uid][i]