from abc import ABCMeta
from abc import abstractmethod
from ray.util.collective.types import (
@property
def world_size(self):
    """Return the number of processes in this group."""
    return self._world_size