from __future__ import annotations
import collections
import itertools
import json
import logging
import math
import os
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, no_type_check
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import FontProperties
from monty.json import MontyDecoder, MSONable
from scipy import interpolate
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from tqdm import tqdm
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.core import DummySpecies, Element, get_el_sp
from pymatgen.core.composition import Composition
from pymatgen.entries import Entry
from pymatgen.util.coord import Simplex, in_coord_list
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import htmlify, latexify
def get_phase_separation_energy(self, entry, **kwargs):
    """
        Provides the energy to the convex hull for the given entry. For stable entries
        already in the phase diagram the algorithm provides the phase separation energy
        which is referred to as the decomposition enthalpy in:

        1. Bartel, C., Trewartha, A., Wang, Q., Dunn, A., Jain, A., Ceder, G.,
            A critical examination of compound stability predictions from
            machine-learned formation energies, npj Computational Materials 6, 97 (2020)

        Args:
            entry (PDEntry): A PDEntry like object
            **kwargs: Keyword args passed to `get_decomp_and_decomp_energy`
                space_limit (int): The maximum number of competing entries to consider.
                stable_only (bool): Only use stable materials as competing entries
                tol (float): The tolerance for convergence of the SLSQP optimization
                    when finding the equilibrium reaction.
                maxiter (int): The maximum number of iterations of the SLSQP optimizer
                    when finding the equilibrium reaction.

        Returns:
            phase separation energy per atom of entry. Stable entries should have
            energies <= 0, Stable elemental entries should have energies = 0 and
            unstable entries should have energies > 0. Entries that have the same
            composition as a stable energy may have positive or negative phase
            separation energies depending on their own energy.
        """
    return self.get_decomp_and_phase_separation_energy(entry, **kwargs)[1]