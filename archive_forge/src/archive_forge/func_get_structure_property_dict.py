from __future__ import annotations
import itertools
import math
import warnings
from typing import TYPE_CHECKING, Literal
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import root
from scipy.special import factorial
from pymatgen.analysis.elasticity.strain import Strain
from pymatgen.analysis.elasticity.stress import Stress
from pymatgen.core.tensors import DEFAULT_QUAD, SquareTensor, Tensor, TensorCollection, get_uvec
from pymatgen.core.units import Unit
from pymatgen.util.due import Doi, due
def get_structure_property_dict(self, structure: Structure, include_base_props: bool=True, ignore_errors: bool=False) -> dict[str, float | Structure | None]:
    """
        Returns a dictionary of properties derived from the elastic tensor
        and an associated structure.

        Args:
            structure (Structure): structure object for which to calculate
                associated properties
            include_base_props (bool): whether to include base properties,
                like k_vrh, etc.
            ignore_errors (bool): if set to true, will set problem properties
                that depend on a physical tensor to None, defaults to False
        """
    s_props = ('trans_v', 'long_v', 'snyder_ac', 'snyder_opt', 'snyder_total', 'clarke_thermalcond', 'cahill_thermalcond', 'debye_temperature')
    sp_dict: dict[str, float | Structure | None]
    if ignore_errors and (self.k_vrh < 0 or self.g_vrh < 0):
        sp_dict = dict.fromkeys(s_props)
    else:
        sp_dict = {prop: getattr(self, prop)(structure) for prop in s_props}
    sp_dict['structure'] = structure
    if include_base_props:
        sp_dict.update(self.property_dict)
    return sp_dict