from __future__ import annotations
import contextlib
import dataclasses
import enum
import functools
import logging
import threading
import traceback
import unittest.mock
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import (
import torch
from torch.utils import _pytree as pytree
from torch.utils._traceback import CapturedTraceback
from torch.utils.weak import WeakTensorKeyDictionary
@staticmethod
@contextlib.contextmanager
def current_frame(frame_summary):
    tc = TracingContext.get()
    if frame_summary is not None:
        tc.frame_summary_stack.append(frame_summary)
    old = tc.loc_in_frame
    tc.loc_in_frame = None
    try:
        yield
    except Exception as e:
        if not hasattr(e, 'real_stack'):
            e.real_stack = tc.extract_stack()
        raise
    finally:
        if frame_summary is not None:
            tc.frame_summary_stack.pop()
        tc.loc_in_frame = old