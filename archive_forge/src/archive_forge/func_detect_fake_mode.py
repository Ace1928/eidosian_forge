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
def detect_fake_mode(inputs: Any=None):
    """
    Attempts to "detect" what the current fake mode is.  If there is one ambiently
    available from TracingContext, we preferentially use that.  Otherwise, we
    heuristically detect the fake mode via the following sources, in order of
    priority:

        - Currently active fake mode on stack
        - Fake mode associated with passed in tensors (inputs does not
          have to be flattened)
    """
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    fake_modes = []
    if (context := TracingContext.try_get()):
        fake_mode = context.fake_mode
        if fake_mode is not None:
            fake_modes.append((fake_mode, 'tracing context', 0))
    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
    for i, m in enumerate(reversed(_get_current_dispatch_mode_stack())):
        if isinstance(m, FakeTensorMode):
            fake_modes.append((m, 'active fake mode', i))
    flat_inputs = pytree.tree_leaves(inputs)
    for i, flat_input in enumerate(flat_inputs):
        if isinstance(flat_input, FakeTensor):
            fake_modes.append((flat_input.fake_mode, 'fake tensor input', i))
    if fake_modes:
        fake_mode, desc1, i1 = fake_modes[0]
        for m, desc2, i2 in fake_modes[1:]:
            assert fake_mode is m, f"fake mode ({fake_mode}) from {desc1} {i1} doesn't match mode ({m}) from {desc2} {i2}\n\nfake mode from {desc1} {i1} allocated at:\n{fake_mode.stack}\nfake mode from {desc2} {i2} allocated at:\n{m.stack}"
        return fake_mode
    else:
        return None