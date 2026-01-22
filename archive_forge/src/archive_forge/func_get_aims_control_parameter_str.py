from __future__ import annotations
import gzip
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.io import zopen
from monty.json import MontyDecoder, MSONable
from pymatgen.core import Lattice, Molecule, Structure
def get_aims_control_parameter_str(self, key: str, value: Any, fmt: str) -> str:
    """Get the string needed to add a parameter to the control.in file

        Args:
            key (str): The name of the input flag
            value (Any): The value to be set for the flag
            fmt (str): The format string to apply to the value

        Returns:
            str: The line to add to the control.in file
        """
    if value is None:
        return ''
    return f'{key:35s}{fmt % value}\n'