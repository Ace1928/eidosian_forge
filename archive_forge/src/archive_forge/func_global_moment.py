from __future__ import annotations
from enum import Enum, unique
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
@property
def global_moment(self):
    """Get the magnetic moment defined in an arbitrary global reference frame.

        Returns:
            np.ndarray of length 3
        """
    return self.get_moment()