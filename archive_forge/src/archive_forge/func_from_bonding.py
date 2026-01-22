from __future__ import annotations
import itertools
import re
import warnings
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable
from monty.serialization import loadfn
from ruamel.yaml import YAML
from pymatgen.core import Element, Lattice, Molecule, Structure
from pymatgen.core.operations import SymmOp
from pymatgen.util.io_utils import clean_lines
@classmethod
def from_bonding(cls, molecule: Molecule, bond: bool=True, angle: bool=True, dihedral: bool=True, tol: float=0.1, **kwargs) -> Self:
    """
        Another constructor that creates an instance from a molecule.
        Covalent bonds and other bond-based topologies (angles and
        dihedrals) can be automatically determined. Cannot be used for
        non bond-based topologies, e.g., improper dihedrals.

        Args:
            molecule (Molecule): Input molecule.
            bond (bool): Whether find bonds. If set to False, angle and
                dihedral searching will be skipped. Default to True.
            angle (bool): Whether find angles. Default to True.
            dihedral (bool): Whether find dihedrals. Default to True.
            tol (float): Bond distance tolerance. Default to 0.1.
                Not recommended to alter.
            **kwargs: Other kwargs supported by Topology.
        """
    real_bonds = molecule.get_covalent_bonds(tol=tol)
    bond_list = [list(map(molecule.index, [b.site1, b.site2])) for b in real_bonds]
    if not all((bond, bond_list)):
        return cls(sites=molecule, **kwargs)
    angle_list, dihedral_list = ([], [])
    dests, freq = np.unique(bond_list, return_counts=True)
    hubs = dests[np.where(freq > 1)].tolist()
    bond_arr = np.array(bond_list)
    if len(hubs) > 0:
        hub_spokes = {}
        for hub in hubs:
            ix = np.any(np.isin(bond_arr, hub), axis=1)
            bonds = np.unique(bond_arr[ix]).tolist()
            bonds.remove(hub)
            hub_spokes[hub] = bonds
    dihedral = False if len(bond_list) < 3 or len(hubs) < 2 else dihedral
    angle = False if len(bond_list) < 2 or len(hubs) < 1 else angle
    if angle:
        for k, v in hub_spokes.items():
            angle_list.extend([[i, k, j] for i, j in itertools.combinations(v, 2)])
    if dihedral:
        hub_cons = bond_arr[np.all(np.isin(bond_arr, hubs), axis=1)]
        for ii, jj in hub_cons.tolist():
            ks = [ki for ki in hub_spokes[ii] if ki != jj]
            ls = [li for li in hub_spokes[jj] if li != ii]
            dihedral_list.extend([[ki, ii, jj, li] for ki, li in itertools.product(ks, ls) if ki != li])
    topologies = {k: v for k, v in zip(SECTION_KEYWORDS['topology'][:3], [bond_list, angle_list, dihedral_list]) if len(v) > 0} or None
    return cls(sites=molecule, topologies=topologies, **kwargs)