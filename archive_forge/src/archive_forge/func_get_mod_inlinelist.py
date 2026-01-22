import _collections_abc
import _weakrefset
import abc
import collections
import contextlib
import copy
import copyreg
import dataclasses
import enum
import functools
import importlib
import inspect
import linecache
import logging
import multiprocessing
import operator
import os
import posixpath
import random
import re
import selectors
import signal
import tempfile
import threading
import tokenize
import traceback
import types
import typing
import unittest
import weakref
from typing import Optional
import torch
import torch._inductor.test_operators
import torch.distributed
import torch.utils._content_store
from .utils import getfile
from .variables.functions import (
@functools.lru_cache(None)
def get_mod_inlinelist():
    inlinelist = set()
    for m in MOD_INLINELIST:
        inlinelist.add(_module_dir(torch) + m[len('torch.'):].replace('.', '/'))
    return inlinelist