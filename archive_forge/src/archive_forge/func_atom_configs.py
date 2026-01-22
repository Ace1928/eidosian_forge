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
def atom_configs(self) -> list[Structure]:
    """Returns AtomConfig object for structures contained in MOVEMENT.

        Returns:
            list[Structure]: List of Structure objects for the structure at each ionic step.
        """
    return [step['atom_config'] for step in self.ionic_steps]