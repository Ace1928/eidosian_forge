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
def get_marker_props(coords, entries):
    """Method for getting marker locations, hovertext, and error bars
            from pd_plot_data.
            """
    x, y, z, texts, energies, uncertainties = ([], [], [], [], [], [])
    is_stable = [entry in self._pd.stable_entries for entry in entries]
    for coord, entry, stable in zip(coords, entries, is_stable):
        energy = round(self._pd.get_form_energy_per_atom(entry), 3)
        entry_id = getattr(entry, 'entry_id', 'no ID')
        comp = entry.composition
        if hasattr(entry, 'original_entry'):
            orig_entry = entry.original_entry
            comp = orig_entry.composition
            entry_id = getattr(orig_entry, 'entry_id', 'no ID')
        formula = comp.reduced_formula
        clean_formula = htmlify(formula)
        label = f'{clean_formula} ({entry_id}) <br> {energy} eV/atom'
        if not stable:
            e_above_hull = round(self._pd.get_e_above_hull(entry), 3)
            if e_above_hull > self.show_unstable:
                continue
            label += f' ({e_above_hull:+} eV/atom)'
            energies.append(e_above_hull)
        else:
            uncertainty = 0
            label += ' (Stable)'
            if hasattr(entry, 'correction_uncertainty_per_atom') and label_uncertainties:
                uncertainty = round(entry.correction_uncertainty_per_atom, 4)
                label += f'<br> (Error: +/- {uncertainty} eV/atom)'
            uncertainties.append(uncertainty)
            energies.append(energy)
        texts.append(label)
        if self._dim == 3 and self.ternary_style == '2d':
            for el, axis in zip(self._pd.elements, [x, y, z]):
                axis.append(entry.composition[el])
        else:
            x.append(coord[0])
            y.append(coord[1])
            if self._dim == 3:
                z.append(energy)
            elif self._dim == 4:
                z.append(coord[2])
    return {'x': x, 'y': y, 'z': z, 'texts': texts, 'energies': energies, 'uncertainties': uncertainties}