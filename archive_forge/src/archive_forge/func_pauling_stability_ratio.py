from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
@property
def pauling_stability_ratio(self):
    """Returns the theoretical Pauling stability ratio (rC/rA) for this environment."""
    if self._pauling_stability_ratio is None:
        if self.ce_symbol in ['S:1', 'L:2']:
            self._pauling_stability_ratio = 0.0
        else:
            min_dist_anions = 1000000
            min_dist_cation_anion = 1000000
            for ipt1 in range(len(self.points)):
                pt1 = np.array(self.points[ipt1])
                min_dist_cation_anion = min(min_dist_cation_anion, np.linalg.norm(pt1 - self.central_site))
                for ipt2 in range(ipt1 + 1, len(self.points)):
                    pt2 = np.array(self.points[ipt2])
                    min_dist_anions = min(min_dist_anions, np.linalg.norm(pt1 - pt2))
            anion_radius = min_dist_anions / 2
            cation_radius = min_dist_cation_anion - anion_radius
            self._pauling_stability_ratio = cation_radius / anion_radius
    return self._pauling_stability_ratio