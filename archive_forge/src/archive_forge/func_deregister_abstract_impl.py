import contextlib
import functools
import warnings
from typing import Callable, Optional
import torch
from torch._library.utils import Kernel, RegistrationHandle
def deregister_abstract_impl():
    if self.lib:
        self.lib._destroy()
        self.lib = None
    self.kernel = None