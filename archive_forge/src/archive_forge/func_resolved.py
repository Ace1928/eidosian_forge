import logging
import sys
from collections import namedtuple
from typing import Optional
import ray
import ray._private.ray_constants as ray_constants
def resolved(self):
    """Returns if this ResourceSpec has default values filled out."""
    for v in self._asdict().values():
        if v is None:
            return False
    return True