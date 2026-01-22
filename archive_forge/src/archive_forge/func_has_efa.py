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
def has_efa() -> bool:
    """
    If shell command `fi_info -p efa -t FI_EP_RDM` returns exit code 0 then we assume that the machine has
    Libfabric EFA interfaces and EFA software components installed,
    see https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html.
    """
    global EFA_PROBE_RESULT
    if EFA_PROBE_RESULT is not None:
        return EFA_PROBE_RESULT
    try:
        EFA_PROBE_RESULT = subprocess.run(['fi_info', '-p', 'efa', '-t', 'FI_EP_RDM'], check=False).returncode == 0
    except FileNotFoundError:
        EFA_PROBE_RESULT = False
    return EFA_PROBE_RESULT