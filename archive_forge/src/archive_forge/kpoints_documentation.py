import re
import warnings
from typing import Dict
import numpy as np
import ase  # Annotations
from ase.utils import jsonable
from ase.cell import Cell
Return band structure of free electrons for this bandpath.

        Keyword arguments are passed to
        :class:`~ase.calculators.test.FreeElectrons`.

        This is for mostly testing and visualization.