from __future__ import annotations
import copy
import itertools
import logging
import math
import typing
import warnings
from collections import Counter
from typing import TYPE_CHECKING, Literal, cast, no_type_check
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import palettable
import scipy.interpolate as scint
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from monty.dev import requires
from monty.json import jsanitize
from pymatgen.core import Element
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.boltztrap import BoltztrapError
from pymatgen.electronic_structure.core import OrbitalType, Spin
from pymatgen.util.plotting import add_fig_kwargs, get_ax3d_fig, pretty_plot
def orbital_label(list_orbitals):
    divide = {}
    for orb in list_orbitals:
        if orb[0] in divide:
            divide[orb[0]].append(orb)
        else:
            divide[orb[0]] = []
            divide[orb[0]].append(orb)
    label = ''
    for elem, orbs in divide.items():
        if elem == 's':
            label += 's,'
        elif len(orbs) == len(individual_orbs[elem]):
            label += elem + ','
        else:
            orb_label = [orb[1:] for orb in orbs]
            label += f'{elem}{str(orb_label).replace('[', '').replace(']', '').replace(', ', '-')},'
    return label[:-1]