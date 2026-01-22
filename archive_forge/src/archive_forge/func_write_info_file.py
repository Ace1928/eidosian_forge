import base64
import dataclasses
import datetime
import errno
import json
import os
import subprocess
import tempfile
import time
import typing
from typing import Optional
from tensorboard import version
from tensorboard.util import tb_logging
def write_info_file(tensorboard_info):
    """Write TensorBoardInfo to the current process's info file.

    This should be called by `main` once the server is ready. When the
    server shuts down, `remove_info_file` should be called.

    Args:
      tensorboard_info: A valid `TensorBoardInfo` object.

    Raises:
      ValueError: If any field on `info` is not of the correct type.
    """
    payload = '%s\n' % _info_to_string(tensorboard_info)
    with open(_get_info_file_path(), 'w') as outfile:
        outfile.write(payload)