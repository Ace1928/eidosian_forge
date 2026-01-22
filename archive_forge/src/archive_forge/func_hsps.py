from __future__ import annotations
import copy
import linecache
from io import StringIO
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.io.pwmat.inputs import ACstrExtractor, AtomConfig, LineLocator
@property
def hsps(self) -> dict[str, np.ndarray]:
    """Return the high symmetry points.

        Returns:
            dict[str, np.ndarray]: The label and fractional coordinate of
                high symmetry points. Return empty dict when task is not
                line-mode kpath.
        """
    return self._hsps