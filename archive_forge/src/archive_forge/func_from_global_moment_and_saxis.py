from __future__ import annotations
from enum import Enum, unique
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
@classmethod
def from_global_moment_and_saxis(cls, global_moment, saxis) -> Self:
    """Convenience method to initialize Magmom from a given global
        magnetic moment, i.e. magnetic moment with saxis=(0,0,1), and
        provided saxis.

        Method is useful if you do not know the components of your
        magnetic moment in frame of your desired saxis.

        Args:
            global_moment: global magnetic moment
            saxis: desired saxis
        """
    magmom = Magmom(global_moment)
    return cls(magmom.get_moment(saxis=saxis), saxis=saxis)