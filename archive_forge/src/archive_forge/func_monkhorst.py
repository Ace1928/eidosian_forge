from __future__ import annotations
import abc
from collections import namedtuple
from collections.abc import Iterable
from enum import Enum, unique
from pprint import pformat
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.collections import AttrDict
from monty.design_patterns import singleton
from monty.json import MontyDecoder, MontyEncoder, MSONable
from pymatgen.core import ArrayWithUnit, Lattice, Species, Structure, units
@classmethod
def monkhorst(cls, ngkpt, shiftk=(0.5, 0.5, 0.5), chksymbreak=None, use_symmetries=True, use_time_reversal=True, comment=None):
    """
        Convenient static constructor for a Monkhorst-Pack mesh.

        Args:
            ngkpt: Subdivisions N_1, N_2 and N_3 along reciprocal lattice vectors.
            shiftk: Shift to be applied to the kpoints.
            use_symmetries: Use spatial symmetries to reduce the number of k-points.
            use_time_reversal: Use time-reversal symmetry to reduce the number of k-points.

        Returns:
            KSampling object.
        """
    return cls(kpts=[ngkpt], kpt_shifts=shiftk, use_symmetries=use_symmetries, use_time_reversal=use_time_reversal, chksymbreak=chksymbreak, comment=comment or 'Monkhorst-Pack scheme with user-specified shiftk')