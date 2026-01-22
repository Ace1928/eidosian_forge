from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from pymatgen.core import Structure
from pymatgen.io.aims.sets.base import AimsInputGenerator
def get_parameter_updates(self, structure: Structure | Molecule, prev_parameters: dict[str, Any]) -> dict:
    """Get the parameter updates for the calculation.

        Parameters
        ----------
        structure: Structure or Molecule
            The structure to calculate the bands for
        prev_parameters: Dict[str, Any]
            The previous parameters

        Returns:
            dict: The updated for the parameters for the output section of FHI-aims
        """
    return {'use_pimd_wrapper': (self.host, self.port)}