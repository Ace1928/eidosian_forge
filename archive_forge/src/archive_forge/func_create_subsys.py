from __future__ import annotations
import itertools
import os
import warnings
import numpy as np
from ruamel.yaml import YAML
from pymatgen.core import SETTINGS
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Element, Molecule, Structure
from pymatgen.io.cp2k.inputs import (
from pymatgen.io.cp2k.utils import get_truncated_coulomb_cutoff, get_unique_site_indices
from pymatgen.io.vasp.inputs import Kpoints as VaspKpoints
from pymatgen.io.vasp.inputs import KpointsSupportedModes
def create_subsys(self, structure: Structure | Molecule) -> None:
    """Create the structure for the input."""
    subsys = Subsys()
    if isinstance(structure, Structure):
        subsys.insert(Cell(structure.lattice))
    else:
        x = max(*structure.cart_coords[:, 0], 1)
        y = max(*structure.cart_coords[:, 1], 1)
        z = max(*structure.cart_coords[:, 2], 1)
        cell = Cell(lattice=Lattice([[10 * x, 0, 0], [0, 10 * y, 0], [0, 0, 10 * z]]))
        cell.add(Keyword('PERIODIC', 'NONE'))
        subsys.insert(cell)
    unique_kinds = get_unique_site_indices(structure)
    for k, v in unique_kinds.items():
        kind = k.split('_')[0]
        kwargs = {}
        _ox = self.structure.site_properties['oxi_state'][v[0]] if 'oxi_state' in self.structure.site_properties else 0
        _sp = self.structure.site_properties['spin'][v[0]] if 'spin' in self.structure.site_properties else 0
        bs = BrokenSymmetry.from_el(kind, _ox, _sp) if _ox else None
        if 'magmom' in self.structure.site_properties and (not bs):
            kwargs['magnetization'] = self.structure.site_properties['magmom'][v[0]]
        if 'ghost' in self.structure.site_properties:
            kwargs['ghost'] = self.structure.site_properties['ghost'][v[0]]
        if 'basis_set' in self.structure.site_properties:
            basis_set = self.structure.site_properties['basis_set'][v[0]]
        else:
            basis_set = self.basis_and_potential[kind].get('basis')
        if 'potential' in self.structure.site_properties:
            potential = self.structure.site_properties['potential'][v[0]]
        else:
            potential = self.basis_and_potential[kind].get('potential')
        if 'aux_basis' in self.structure.site_properties:
            kwargs['aux_basis'] = self.structure.site_properties['aux_basis'][v[0]]
        elif self.basis_and_potential[kind].get('aux_basis'):
            kwargs['aux_basis'] = self.basis_and_potential[kind].get('aux_basis')
        _kind = Kind(kind, alias=k, basis_set=basis_set, potential=potential, subsections={'BS': bs} if bs else {}, **kwargs)
        if self.qs_method.upper() == 'GAPW':
            _kind.add(Keyword('RADIAL_GRID', 200))
            _kind.add(Keyword('LEBEDEV_GRID', 80))
        subsys.insert(_kind)
    coord = Coord(structure, aliases=unique_kinds)
    subsys.insert(coord)
    self['FORCE_EVAL'].insert(subsys)