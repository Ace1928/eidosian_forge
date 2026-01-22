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
def get_pd_for_entry(self, entry: Entry | Composition) -> PhaseDiagram:
    """
        Get the possible phase diagrams for an entry.

        Args:
            entry (PDEntry | Composition): A PDEntry or Composition-like object

        Returns:
            PhaseDiagram: phase diagram that the entry is part of

        Raises:
            ValueError: If no suitable PhaseDiagram is found for the entry.
        """
    entry_space = frozenset(entry.elements) if isinstance(entry, Composition) else frozenset(entry.elements)
    try:
        return self.pds[entry_space]
    except KeyError:
        for space, pd in self.pds.items():
            if space.issuperset(entry_space):
                return pd
    raise ValueError(f'No suitable PhaseDiagrams found for {entry}.')