from __future__ import annotations
import abc
import time
from .logs import logger, null_logger
from typing import Optional, List, Dict, Any, Union
@property
def data_dict(self) -> Dict[str, Any]:
    """
        Returns a dict representation of the timer
        """
    return {'duration': self.duration, 'total': self.total, 'elapsed': self.elapsed, 'average': self.average, 'data': self.dformat(self.total)}