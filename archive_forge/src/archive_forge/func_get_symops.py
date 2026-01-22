from __future__ import annotations
import math
import os
import re
import textwrap
import warnings
from collections import defaultdict, deque
from functools import partial
from inspect import getfullargspec
from io import StringIO
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.serialization import loadfn
from pymatgen.core import Composition, DummySpecies, Element, Lattice, PeriodicSite, Species, Structure, get_el_sp
from pymatgen.core.operations import MagSymmOp, SymmOp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SpacegroupOperations
from pymatgen.symmetry.groups import SYMM_DATA, SpaceGroup
from pymatgen.symmetry.maggroups import MagneticSpaceGroup
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list_pbc, in_coord_list_pbc
def get_symops(self, data):
    """
        In order to generate symmetry equivalent positions, the symmetry
        operations are parsed. If the symops are not present, the space
        group symbol is parsed, and symops are generated.
        """
    sym_ops = []
    for symmetry_label in ['_symmetry_equiv_pos_as_xyz', '_symmetry_equiv_pos_as_xyz_', '_space_group_symop_operation_xyz', '_space_group_symop_operation_xyz_']:
        if data.data.get(symmetry_label):
            xyz = data.data.get(symmetry_label)
            if isinstance(xyz, str):
                msg = 'A 1-line symmetry op P1 CIF is detected!'
                warnings.warn(msg)
                self.warnings.append(msg)
                xyz = [xyz]
            try:
                sym_ops = [SymmOp.from_xyz_str(s) for s in xyz]
                break
            except ValueError:
                continue
    if not sym_ops:
        for symmetry_label in ['_symmetry_space_group_name_H-M', '_symmetry_space_group_name_H_M', '_symmetry_space_group_name_H-M_', '_symmetry_space_group_name_H_M_', '_space_group_name_Hall', '_space_group_name_Hall_', '_space_group_name_H-M_alt', '_space_group_name_H-M_alt_', '_symmetry_space_group_name_hall', '_symmetry_space_group_name_hall_', '_symmetry_space_group_name_h-m', '_symmetry_space_group_name_h-m_']:
            sg = data.data.get(symmetry_label)
            msg_template = 'No _symmetry_equiv_pos_as_xyz type key found. Spacegroup from {} used.'
            if sg:
                sg = sub_space_group(sg)
                try:
                    spg = space_groups.get(sg)
                    if spg:
                        sym_ops = SpaceGroup(spg).symmetry_ops
                        msg = msg_template.format(symmetry_label)
                        warnings.warn(msg)
                        self.warnings.append(msg)
                        break
                except ValueError:
                    pass
                try:
                    cod_data = loadfn(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'symmetry', 'symm_ops.json'))
                    for d in cod_data:
                        if sg == re.sub('\\s+', '', d['hermann_mauguin']):
                            xyz = d['symops']
                            sym_ops = [SymmOp.from_xyz_str(s) for s in xyz]
                            msg = msg_template.format(symmetry_label)
                            warnings.warn(msg)
                            self.warnings.append(msg)
                            break
                except Exception:
                    continue
                if sym_ops:
                    break
    if not sym_ops:
        for symmetry_label in ['_space_group_IT_number', '_space_group_IT_number_', '_symmetry_Int_Tables_number', '_symmetry_Int_Tables_number_']:
            if data.data.get(symmetry_label):
                try:
                    i = int(str2float(data.data.get(symmetry_label)))
                    sym_ops = SpaceGroup.from_int_number(i).symmetry_ops
                    break
                except ValueError:
                    continue
    if not sym_ops:
        msg = 'No _symmetry_equiv_pos_as_xyz type key found. Defaulting to P1.'
        warnings.warn(msg)
        self.warnings.append(msg)
        sym_ops = [SymmOp.from_xyz_str(s) for s in ['x', 'y', 'z']]
    return sym_ops