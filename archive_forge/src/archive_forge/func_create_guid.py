import os as _os
import sys as _sys
import warnings as _warnings
from .base import Sign
from .controller_db import mapping_list
def create_guid(bus: int, vendor: int, product: int, version: int, name: str, signature: int, data: int) -> str:
    """Create an SDL2 style GUID string from a device's identifiers."""
    bus = _swap_le16(bus)
    vendor = _swap_le16(vendor)
    product = _swap_le16(product)
    version = _swap_le16(version)
    return f'{bus:04x}0000{vendor:04x}0000{product:04x}0000{version:04x}0000'