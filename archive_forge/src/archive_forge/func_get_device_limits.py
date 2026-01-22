import math
from dataclasses import dataclass, field
from typing import Mapping, Tuple
import torch
def get_device_limits(device) -> DeviceLimit:
    """Currently only implemented for GPUs"""
    if device is not None and device.type == 'cuda':
        device_sm = torch.cuda.get_device_capability(device)
        device_name = torch.cuda.get_device_name(device)
        for lim in DEVICE_LIMITS:
            if lim.sm == device_sm:
                if lim.name in device_name:
                    return lim
    return DeviceLimit()