import inspect
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union
import torch
from torch._streambase import _EventBase, _StreamBase
def register_interface_for_device(device: str, device_interface: Type[DeviceInterface]):
    device_interfaces[device] = device_interface