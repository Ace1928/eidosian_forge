from __future__ import annotations
import collections
import contextlib
import enum
import functools
import getpass
import inspect
import itertools
import logging
import math
import operator
import os
import platform
import re
import shutil
import sys
import tempfile
import textwrap
import time
import unittest
from io import StringIO
from typing import (
from unittest import mock
import sympy
from typing_extensions import Concatenate, ParamSpec
import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch.autograd import DeviceType
from torch.autograd.profiler_util import EventList
from torch.utils._sympy.functions import CeilDiv, CleanDiv, FloorDiv, ModularIndexing
from . import config
def create_bandwidth_info_str(ms, num_gb, gb_per_s, prefix='', suffix=''):
    info_str = f'{prefix}{ms:.3f}ms    \t{num_gb:.3f} GB \t {gb_per_s:7.2f}GB/s{suffix}'
    try:
        import colorama
        if ms > 0.012 and gb_per_s < 650:
            info_str = colorama.Fore.RED + info_str + colorama.Fore.RESET
    except ImportError:
        log.warning('Colorama is not installed. Install it if you want colored output')
    return info_str