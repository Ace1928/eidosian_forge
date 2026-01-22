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
def init_pool_generator(gens, random_seed=None, id_queue=None):
    """Initializer function for pool workers.

  Args:
    gens: State which should be made available to worker processes.
    random_seed: An optional value with which to seed child processes.
    id_queue: A multiprocessing Queue of worker ids. This is used to indicate
      that a worker process was created by Keras and can be terminated using
      the cleanup_all_keras_forkpools utility.
  """
    global _SHARED_SEQUENCES
    _SHARED_SEQUENCES = gens
    worker_proc = multiprocessing.current_process()
    worker_proc.name = 'Keras_worker_{}'.format(worker_proc.name)
    if random_seed is not None:
        np.random.seed(random_seed + worker_proc.ident)
    if id_queue is not None:
        id_queue.put(worker_proc.ident, block=True, timeout=0.1)