from __future__ import annotations
import abc
import time
import atexit
import contextlib
from lazyops.libs import lazyload
from pydantic import BaseModel
from pydantic.types import ByteSize
from lazyops.utils import logger, Timer
from typing import Optional, List, Dict, Any, Union, Type, Set, TYPE_CHECKING
def get_memory(self) -> Dict[str, float]:
    """
        Returns the Memory Usage
        """
    mem = psutil.virtual_memory()
    return {'total': mem.total, 'available': mem.available, 'percent': mem.percent, 'used': mem.used, 'free': mem.free, 'active': mem.active, 'inactive': mem.inactive}