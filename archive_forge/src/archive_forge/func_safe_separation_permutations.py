from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
def safe_separation_permutations(self, ordered_plane=False, ordered_point_groups=None, add_opposite=False):
    """
        Simple and safe permutations for this separation plane.

        This is not meant to be used in production. Default configuration for ChemEnv does not use this method.

        Args:
            ordered_plane: Whether the order of the points in the plane can be used to reduce the
                number of permutations.
            ordered_point_groups: Whether the order of the points in each point group can be used to reduce the
                number of permutations.
            add_opposite: Whether to add the permutations from the second group before the first group as well.

        Returns:
            list[int]: safe permutations.
        """
    s0 = list(range(len(self.point_groups[0])))
    plane = list(range(len(self.point_groups[0]), len(self.point_groups[0]) + len(self.plane_points)))
    s1 = list(range(len(self.point_groups[0]) + len(self.plane_points), len(self.point_groups[0]) + len(self.plane_points) + len(self.point_groups[1])))
    ordered_point_groups = [False, False] if ordered_point_groups is None else ordered_point_groups

    def rotate(s, n):
        return s[-n:] + s[:-n]
    if ordered_plane and self.ordered_plane:
        plane_perms = [rotate(plane, ii) for ii in range(len(plane))]
        inv_plane = plane[::-1]
        plane_perms.extend([rotate(inv_plane, ii) for ii in range(len(inv_plane))])
    else:
        plane_perms = list(itertools.permutations(plane))
    if ordered_point_groups[0] and self.ordered_point_groups[0]:
        s0_perms = [rotate(s0, ii) for ii in range(len(s0))]
        inv_s0 = s0[::-1]
        s0_perms.extend([rotate(inv_s0, ii) for ii in range(len(inv_s0))])
    else:
        s0_perms = list(itertools.permutations(s0))
    if ordered_point_groups[1] and self.ordered_point_groups[1]:
        s1_perms = [rotate(s1, ii) for ii in range(len(s1))]
        inv_s1 = s1[::-1]
        s1_perms.extend([rotate(inv_s1, ii) for ii in range(len(inv_s1))])
    else:
        s1_perms = list(itertools.permutations(s1))
    if self._safe_permutations is None:
        self._safe_permutations = []
        for perm_side1 in s0_perms:
            for perm_sep_plane in plane_perms:
                for perm_side2 in s1_perms:
                    perm = list(perm_side1)
                    perm.extend(list(perm_sep_plane))
                    perm.extend(list(perm_side2))
                    self._safe_permutations.append(perm)
                    if add_opposite:
                        perm = list(perm_side2)
                        perm.extend(list(perm_sep_plane))
                        perm.extend(list(perm_side1))
                        self._safe_permutations.append(perm)
    return self._safe_permutations