from abc import ABC, abstractmethod
import contextlib
from typing import Any
import torch
import torch.utils._pytree as pytree
from torch._C._functorch import (
from torch.autograd.forward_ad import _set_fwd_grad_enabled

This file contains the functorch integration with PyDispatcher.

PyDispatcher does not understand functorch's DynamicLayerStack dispatching
logic because it is entirely implemented in C++ in the fallbacks for two
dispatch keys, FuncTorchDynamicLayer{Front, Back}Mode (PyDispatcher is unable
to directly reuse C++ boxed fallbacks).

Instead of trying to hammer PyDispatcher into understanding those fallbacks,
we re-implement the logic of peeking the top of the stack for an interpreter,
selecting the interpreter to dispatch on, etc, in Python. This leads to a
simpler design.

The main difference between C++ functorch and PyDispatcher's functorch logic
is that:
- C++ functorch needs to manually tweak dispatch keys to ping-pong between
  DynamicLayerFrontMode and DynamicLayerBackMode.
- PyDispatcher's functorch logic pops an Interpreter from the top of the stack
  and asks it to execute the rule associated with the Interpreter.

In C++ we do the ping-pong because e.g. vmap rules are associated with the
batched DispatchKey, but in PyDispatcher we are able to avoid this by asking
the user to register a batching rule directly to a transform that an
interpreter then invokes.
