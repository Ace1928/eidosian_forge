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
def get_decomp_and_e_above_hull(self, entry: PDEntry, allow_negative: bool=False, check_stable: bool=False, on_error: Literal['raise', 'warn', 'ignore']='raise') -> tuple[dict[PDEntry, float], float] | tuple[None, None]:
    """Same as method on parent class PhaseDiagram except check_stable defaults to False
        for speed. See https://github.com/materialsproject/pymatgen/issues/2840 for details.
        """
    return super().get_decomp_and_e_above_hull(entry=entry, allow_negative=allow_negative, check_stable=check_stable, on_error=on_error)