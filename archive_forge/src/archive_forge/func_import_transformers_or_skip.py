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
def import_transformers_or_skip():

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                from transformers import AutoModelForMaskedLM, BertConfig
                return func(*args, **kwargs)
            except ImportError:
                sys.exit(TEST_SKIPS['importerror'].exit_code)
        return wrapper
    return decorator