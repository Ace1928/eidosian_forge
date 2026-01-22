from __future__ import annotations
import os
from monty.design_patterns import singleton
from pymatgen.core import Composition, Element

        Gets a designation for low, medium, high HHI, as specified in "U.S.
        Department of Justice and the Federal Trade Commission, Horizontal
        merger guidelines; 2010.".

        Args:
            hhi (float): HHI value

        Returns:
            The designation as String
        