from __future__ import annotations
import collections
import itertools
from math import acos, pi
from typing import TYPE_CHECKING
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
from pymatgen.analysis.local_env import JmolNN, VoronoiNN
from pymatgen.core import Composition, Element, PeriodicSite, Species
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_percentage_volume_change(self) -> float:
    """
        Returns the percentage volume change.

        Returns:
            float: Volume change in percent. 0.055 means a 5.5% increase.
        """
    return self.final.volume / self.initial.volume - 1