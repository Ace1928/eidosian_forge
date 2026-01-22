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
def run_and_get_code(fn, *args, **kwargs):
    from .graph import GraphLowering
    compile_to_module = GraphLowering.compile_to_module
    source_codes = []

    def patched_compile_to_module(self):
        mod = compile_to_module(self)
        with open(mod.__file__) as f:
            source_codes.append(f.read())
        return mod
    with mock.patch.object(GraphLowering, 'compile_to_module', patched_compile_to_module):
        torch._dynamo.reset()
        result = fn(*args, **kwargs)
    return (result, source_codes)