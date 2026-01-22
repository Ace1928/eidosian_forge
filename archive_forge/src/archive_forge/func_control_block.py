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
@property
def control_block(self) -> str:
    """Get the block of text for the control.in file of the Cube"""
    cb = f'output cube {self.type}\n'
    cb += f'    cube origin {self.origin[0]: .12e} {self.origin[1]: .12e} {self.origin[2]: .12e}\n'
    for idx in range(3):
        cb += f'    cube edge {self.points[idx]} '
        cb += f'{self.edges[idx][0]: .12e} '
        cb += f'{self.edges[idx][1]: .12e} '
        cb += f'{self.edges[idx][2]: .12e}\n'
    cb += f'    cube format {self.format}\n'
    if self.spin_state is not None:
        cb += f'    cube spinstate {self.spin_state}\n'
    if self.kpoint is not None:
        cb += f'    cube kpoint {self.kpoint}\n'
    if self.filename is not None:
        cb += f'    cube filename {self.filename}\n'
    if self.elf_type is not None:
        cb += f'    cube elf_type {self.elf_type}\n'
    return cb