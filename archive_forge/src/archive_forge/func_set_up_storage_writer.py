import abc
from dataclasses import dataclass
from typing import List, Any
from torch.futures import Future
from .metadata import (
from .planner import (
@abc.abstractmethod
def set_up_storage_writer(self, is_coordinator: bool) -> None:
    """
        Initialize this instance.

        Args:
            is_coordinator (bool): Whether this instance is responsible for coordinating
              the checkpoint.
        """
    pass