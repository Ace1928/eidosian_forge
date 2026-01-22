import base64
import collections
import contextlib
import json
import logging
import os
import pickle
import subprocess
import sys
import time
import timeit
from ._interfaces import Model
import six
from tensorflow.python.framework import dtypes  # pylint: disable=g-direct-tensorflow-import
def copy_model_to_local(gcs_path, dest_path):
    """Copy files from gcs to a local path.

  Copies files directly to the dest_path.
  Sample behavior:
  dir1/
    file1
    file2
    dir2/
      file3

  copy_model_to_local("dir1", "/tmp")
  After copy:
  tmp/
    file1
    file2
    dir2/
      file3

  Args:
    gcs_path: Source GCS path that we're copying from.
    dest_path: Destination local path that we're copying to.

  Raises:
    Exception: If gsutil is not found.
  """
    copy_start_time = time.time()
    logging.debug('Starting to copy files from %s to %s', gcs_path, dest_path)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    gcs_path = os.path.join(gcs_path, '*')
    try:
        subprocess.check_call(['gsutil', 'cp', '-R', gcs_path, dest_path], stdin=subprocess.PIPE)
    except subprocess.CalledProcessError:
        logging.exception('Could not copy model using gsutil.')
        raise
    logging.debug('Files copied from %s to %s: took %f seconds', gcs_path, dest_path, time.time() - copy_start_time)