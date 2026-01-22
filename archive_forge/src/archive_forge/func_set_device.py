import inspect
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union
import torch
from torch._streambase import _EventBase, _StreamBase
@staticmethod
def set_device(device: int):
    caching_worker_current_devices['cuda'] = device