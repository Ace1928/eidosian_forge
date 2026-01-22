from __future__ import annotations
import copy
import itertools
import logging
import math
import warnings
from collections import defaultdict
from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from math import cos, sin
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
import scipy.cluster
import spglib
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Molecule, PeriodicSite, Structure
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list, pbc_diff
from pymatgen.util.due import Doi, due
@cite_conventional_cell_algo
def get_conventional_standard_structure(self, international_monoclinic=True, keep_site_properties=False):
    """Gives a structure with a conventional cell according to certain standards. The
        standards are defined in Setyawan, W., & Curtarolo, S. (2010). High-throughput
        electronic band structure calculations: Challenges and tools. Computational
        Materials Science, 49(2), 299-312. doi:10.1016/j.commatsci.2010.05.010 They
        basically enforce as much as possible norm(a1)<norm(a2)<norm(a3). NB This is not
        necessarily the same as the standard settings within the International Tables of
        Crystallography, for which get_refined_structure should be used instead.

        Args:
            international_monoclinic (bool): Whether to convert to proper international convention
                such that beta is the non-right angle.
            keep_site_properties (bool): Whether to keep the input site properties (including
                magnetic moments) on the sites that are still present after the refinement. Note:
                This is disabled by default because the magnetic moments are not always directly
                transferable between unit cell definitions. For instance, long-range magnetic
                ordering or antiferromagnetic character may no longer be present (or exist in
                the same way) in the returned structure. If keep_site_properties is True,
                each site retains the same site property as in the original structure without
                further adjustment.

        Returns:
            The structure in a conventional standardized cell
        """
    tol = 1e-05
    struct = self.get_refined_structure(keep_site_properties=keep_site_properties)
    lattice = struct.lattice
    latt_type = self.get_lattice_type()
    sorted_lengths = sorted(lattice.abc)
    sorted_dic = sorted(({'vec': lattice.matrix[i], 'length': lattice.abc[i], 'orig_index': i} for i in range(3)), key=lambda k: k['length'])
    if latt_type in ('orthorhombic', 'cubic'):
        transf = np.zeros(shape=(3, 3))
        if self.get_space_group_symbol().startswith('C'):
            transf[2] = [0, 0, 1]
            a, b = sorted(lattice.abc[:2])
            sorted_dic = sorted(({'vec': lattice.matrix[i], 'length': lattice.abc[i], 'orig_index': i} for i in [0, 1]), key=lambda k: k['length'])
            for idx in range(2):
                transf[idx][sorted_dic[idx]['orig_index']] = 1
            c = lattice.abc[2]
        elif self.get_space_group_symbol().startswith('A'):
            transf[2] = [1, 0, 0]
            a, b = sorted(lattice.abc[1:])
            sorted_dic = sorted(({'vec': lattice.matrix[i], 'length': lattice.abc[i], 'orig_index': i} for i in [1, 2]), key=lambda k: k['length'])
            for idx in range(2):
                transf[idx][sorted_dic[idx]['orig_index']] = 1
            c = lattice.abc[0]
        else:
            for idx, dct in enumerate(sorted_dic):
                transf[idx][dct['orig_index']] = 1
            a, b, c = sorted_lengths
        lattice = Lattice.orthorhombic(a, b, c)
    elif latt_type == 'tetragonal':
        transf = np.zeros(shape=(3, 3))
        a, b, c = sorted_lengths
        for idx, dct in enumerate(sorted_dic):
            transf[idx][dct['orig_index']] = 1
        if abs(b - c) < tol < abs(a - c):
            a, c = (c, a)
            transf = np.dot([[0, 0, 1], [0, 1, 0], [1, 0, 0]], transf)
        lattice = Lattice.tetragonal(a, c)
    elif latt_type in ('hexagonal', 'rhombohedral'):
        a, b, c = lattice.abc
        if np.all(np.abs([a - b, c - b, a - c]) < 0.001):
            struct.make_supercell(((1, -1, 0), (0, 1, -1), (1, 1, 1)))
            a, b, c = sorted(struct.lattice.abc)
        if abs(b - c) < 0.001:
            a, c = (c, a)
        new_matrix = [[a / 2, -a * math.sqrt(3) / 2, 0], [a / 2, a * math.sqrt(3) / 2, 0], [0, 0, c]]
        lattice = Lattice(new_matrix)
        transf = np.eye(3, 3)
    elif latt_type == 'monoclinic':
        if self.get_space_group_operations().int_symbol.startswith('C'):
            transf = np.zeros(shape=(3, 3))
            transf[2] = [0, 0, 1]
            sorted_dic = sorted(({'vec': lattice.matrix[i], 'length': lattice.abc[i], 'orig_index': i} for i in [0, 1]), key=lambda k: k['length'])
            a = sorted_dic[0]['length']
            b = sorted_dic[1]['length']
            c = lattice.abc[2]
            new_matrix = None
            for t in itertools.permutations(list(range(2)), 2):
                m = lattice.matrix
                latt2 = Lattice([m[t[0]], m[t[1]], m[2]])
                lengths = latt2.lengths
                angles = latt2.angles
                if angles[0] > 90:
                    a, b, c, alpha, beta, gamma = Lattice([-m[t[0]], -m[t[1]], m[2]]).parameters
                    transf = np.zeros(shape=(3, 3))
                    transf[0][t[0]] = -1
                    transf[1][t[1]] = -1
                    transf[2][2] = 1
                    alpha = math.pi * alpha / 180
                    new_matrix = [[a, 0, 0], [0, b, 0], [0, c * cos(alpha), c * sin(alpha)]]
                    continue
                if angles[0] < 90:
                    transf = np.zeros(shape=(3, 3))
                    transf[0][t[0]] = 1
                    transf[1][t[1]] = 1
                    transf[2][2] = 1
                    a, b, c = lengths
                    alpha = math.pi * angles[0] / 180
                    new_matrix = [[a, 0, 0], [0, b, 0], [0, c * cos(alpha), c * sin(alpha)]]
            if new_matrix is None:
                new_matrix = [[a, 0, 0], [0, b, 0], [0, 0, c]]
                transf = np.zeros(shape=(3, 3))
                transf[2] = [0, 0, 1]
                for idx, dct in enumerate(sorted_dic):
                    transf[idx][dct['orig_index']] = 1
        else:
            new_matrix = None
            for t in itertools.permutations(list(range(3)), 3):
                m = lattice.matrix
                a, b, c, alpha, beta, gamma = Lattice([m[t[0]], m[t[1]], m[t[2]]]).parameters
                if alpha > 90 and b < c:
                    a, b, c, alpha, beta, gamma = Lattice([-m[t[0]], -m[t[1]], m[t[2]]]).parameters
                    transf = np.zeros(shape=(3, 3))
                    transf[0][t[0]] = -1
                    transf[1][t[1]] = -1
                    transf[2][t[2]] = 1
                    alpha = math.pi * alpha / 180
                    new_matrix = [[a, 0, 0], [0, b, 0], [0, c * cos(alpha), c * sin(alpha)]]
                    continue
                if alpha < 90 and b < c:
                    transf = np.zeros(shape=(3, 3))
                    transf[0][t[0]] = 1
                    transf[1][t[1]] = 1
                    transf[2][t[2]] = 1
                    alpha = math.pi * alpha / 180
                    new_matrix = [[a, 0, 0], [0, b, 0], [0, c * cos(alpha), c * sin(alpha)]]
            if new_matrix is None:
                new_matrix = [[sorted_lengths[0], 0, 0], [0, sorted_lengths[1], 0], [0, 0, sorted_lengths[2]]]
                transf = np.zeros(shape=(3, 3))
                for idx, dct in enumerate(sorted_dic):
                    transf[idx][dct['orig_index']] = 1
        if international_monoclinic:
            op = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
            transf = np.dot(op, transf)
            new_matrix = np.dot(op, new_matrix)
            beta = Lattice(new_matrix).beta
            if beta < 90:
                op = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
                transf = np.dot(op, transf)
                new_matrix = np.dot(op, new_matrix)
        lattice = Lattice(new_matrix)
    elif latt_type == 'triclinic':
        struct = struct.get_reduced_structure('LLL')
        lattice = struct.lattice
        a, b, c = lattice.lengths
        alpha, beta, gamma = (math.pi * i / 180 for i in lattice.angles)
        new_matrix = None
        test_matrix = [[a, 0, 0], [b * cos(gamma), b * sin(gamma), 0.0], [c * cos(beta), c * (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma), c * math.sqrt(sin(gamma) ** 2 - cos(alpha) ** 2 - cos(beta) ** 2 + 2 * cos(alpha) * cos(beta) * cos(gamma)) / sin(gamma)]]

        def is_all_acute_or_obtuse(matrix) -> bool:
            recp_angles = np.array(Lattice(matrix).reciprocal_lattice.angles)
            return all(recp_angles <= 90) or all(recp_angles > 90)
        if is_all_acute_or_obtuse(test_matrix):
            transf = np.eye(3)
            new_matrix = test_matrix
        test_matrix = [[-a, 0, 0], [b * cos(gamma), b * sin(gamma), 0.0], [-c * cos(beta), -c * (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma), -c * math.sqrt(sin(gamma) ** 2 - cos(alpha) ** 2 - cos(beta) ** 2 + 2 * cos(alpha) * cos(beta) * cos(gamma)) / sin(gamma)]]
        if is_all_acute_or_obtuse(test_matrix):
            transf = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
            new_matrix = test_matrix
        test_matrix = [[-a, 0, 0], [-b * cos(gamma), -b * sin(gamma), 0.0], [c * cos(beta), c * (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma), c * math.sqrt(sin(gamma) ** 2 - cos(alpha) ** 2 - cos(beta) ** 2 + 2 * cos(alpha) * cos(beta) * cos(gamma)) / sin(gamma)]]
        if is_all_acute_or_obtuse(test_matrix):
            transf = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
            new_matrix = test_matrix
        test_matrix = [[a, 0, 0], [-b * cos(gamma), -b * sin(gamma), 0.0], [-c * cos(beta), -c * (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma), -c * math.sqrt(sin(gamma) ** 2 - cos(alpha) ** 2 - cos(beta) ** 2 + 2 * cos(alpha) * cos(beta) * cos(gamma)) / sin(gamma)]]
        if is_all_acute_or_obtuse(test_matrix):
            transf = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
            new_matrix = test_matrix
        lattice = Lattice(new_matrix)
    new_coords = np.dot(transf, np.transpose(struct.frac_coords)).T
    new_struct = Structure(lattice, struct.species_and_occu, new_coords, site_properties=struct.site_properties, to_unit_cell=True)
    return new_struct.get_sorted_structure()