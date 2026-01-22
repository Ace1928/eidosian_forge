from abc import ABCMeta
from abc import abstractmethod
from ray.util.collective.types import (
def destroy_group(self):
    """GC the communicators."""
    pass