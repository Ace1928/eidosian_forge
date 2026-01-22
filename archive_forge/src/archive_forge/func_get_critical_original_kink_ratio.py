from __future__ import annotations
import json
import os
import warnings
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
from monty.json import MSONable
from pandas import DataFrame
from plotly.graph_objects import Figure, Scatter
from pymatgen.analysis.phase_diagram import GrandPotentialPhaseDiagram, PhaseDiagram
from pymatgen.analysis.reaction_calculator import Reaction
from pymatgen.core.composition import Composition
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import htmlify, latexify
def get_critical_original_kink_ratio(self):
    """
        Returns a list of molar mixing ratio for each kink between ORIGINAL
        (instead of processed) reactant compositions. This is the
        same list as mixing ratio obtained from get_kinks method
        if self.norm = False.

        Returns:
            A list of floats representing molar mixing ratios between
            the original reactant compositions for each kink.
        """
    ratios = []
    if self.c1_original == self.c2_original:
        return [0, 1]
    reaction_kink = [k[3] for k in self.get_kinks()]
    for rxt in reaction_kink:
        ratios.append(abs(self._get_original_composition_ratio(rxt)))
    return ratios