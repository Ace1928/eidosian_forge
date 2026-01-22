from __future__ import annotations
import os
from monty.design_patterns import singleton
from pymatgen.core import Composition, Element
def get_hhi_production(self, comp_or_form):
    """
        Gets the production HHI for a compound.

        Args:
            comp_or_form (Composition or String): A Composition or String formula

        Returns:
            The HHI production value
        """
    return self.get_hhi(comp_or_form)[0]