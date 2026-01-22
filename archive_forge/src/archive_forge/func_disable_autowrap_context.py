import contextlib
import copy
from abc import ABC, abstractmethod
from typing import (
import torch.nn as nn
@staticmethod
def disable_autowrap_context() -> None:
    _ConfigAutoWrap.in_autowrap_context = False
    _ConfigAutoWrap.wrapper_cls = None
    _ConfigAutoWrap.kwargs = {}