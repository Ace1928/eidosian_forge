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
def set_export_info(self, guard_type, guarded_class, code_list, obj_weakref):
    if not self.guard_types:
        self.guard_types = list()
    self.guard_types.append(guard_type)
    assert self.guarded_class_weakref in (guarded_class, None), 'Guarded class id must be identical, or None'
    self.guarded_class_weakref = guarded_class
    if not self.code_list:
        self.code_list = code_list
    else:
        self.code_list.extend(code_list)
    assert self.obj_weakref in (obj_weakref, None), 'Guarded object must be identical, or None'
    self.obj_weakref = obj_weakref