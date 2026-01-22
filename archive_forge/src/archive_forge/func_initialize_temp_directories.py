import faulthandler
import logging
import multiprocessing
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from functools import partial, reduce, wraps
from io import StringIO
from typing import Dict, NamedTuple, Optional, Union
from unittest.mock import patch
import torch
import torch._dynamo.test_case
import torch.cuda.nccl
import torch.distributed as c10d
import torch.nn as nn
from torch.testing._internal.common_utils import (
from torch.testing._internal.distributed.multi_threaded_pg import (
def initialize_temp_directories(init_method: Optional[str]=None) -> None:
    global tmp_dir
    tmp_dir = tempfile.TemporaryDirectory()
    os.environ['TEMP_DIR'] = tmp_dir.name
    os.mkdir(os.path.join(tmp_dir.name, 'barrier'))
    os.mkdir(os.path.join(tmp_dir.name, 'test_dir'))
    init_dir_path = os.path.join(tmp_dir.name, 'init_dir')
    os.mkdir(init_dir_path)
    if init_method is not None:
        os.environ['INIT_METHOD'] = init_method
    else:
        os.environ['INIT_METHOD'] = FILE_SCHEMA + os.path.join(init_dir_path, 'shared_init_file')