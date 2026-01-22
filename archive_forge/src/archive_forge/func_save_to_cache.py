import errno
import os
import shutil
import tempfile
from typing import Optional
from absl import logging
from pyglib import stringutil
def save_to_cache(cache_root: str, api_name: str, api_version: str, discovery_document: str) -> None:
    """Saves a discovery document to the on-disc cache with key `api` and `version`.

  Args:
    cache_root: [str], a directory where all cache files are stored.
    api_name: [str], Name of api `discovery_document` to be saved.
    api_version: [str], Version of `discovery_document`.
    discovery_document: [str]. Discovery document as a json string.

  Raises:
    OSError: If an error occurs when the file is written.
  """
    directory = os.path.join(cache_root, api_name, api_version)
    file = os.path.join(directory, _DISCOVERY_CACHE_FILE)
    if os.path.isfile(file):
        return
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    tmpdir = tempfile.mkdtemp(dir=directory)
    try:
        temp_file_path = os.path.join(tmpdir, 'tmp.json')
        with open(temp_file_path, 'wb') as f:
            f.write(stringutil.ensure_binary(discovery_document, 'utf8'))
            f.flush()
            os.fsync(f.fileno())
        os.rename(temp_file_path, file)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)