from __future__ import annotations
import abc
from collections import namedtuple
from collections.abc import Iterable
from enum import Enum, unique
from pprint import pformat
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.collections import AttrDict
from monty.design_patterns import singleton
from monty.json import MontyDecoder, MontyEncoder, MSONable
from pymatgen.core import ArrayWithUnit, Lattice, Species, Structure, units
def structure_from_abivars(cls=None, *args, **kwargs) -> Structure:
    """
    Build a Structure object from a dictionary with ABINIT variables.

    Args:
        cls: Structure class to be instantiated. Defaults to Structure.

    Example:
        al_structure = structure_from_abivars(
            acell=3*[7.5],
            rprim=[0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0],
            typat=1,
            xred=[0.0, 0.0, 0.0],
            ntypat=1,
            znucl=13,
        )

    xred can be replaced with xcart or xangst.
    """
    kwargs.update(dict(*args))
    cls = cls or Structure
    lattice = lattice_from_abivars(**kwargs)
    coords, coords_are_cartesian = (kwargs.get('xred'), False)
    if coords is None:
        coords = kwargs.get('xcart')
        if coords is not None:
            if 'xangst' in kwargs:
                raise ValueError('xangst and xcart are mutually exclusive')
            coords = ArrayWithUnit(coords, 'bohr').to('ang')
        else:
            coords = kwargs.get('xangst')
        coords_are_cartesian = True
    if coords is None:
        raise ValueError(f'Cannot extract coordinates from:\n {kwargs}')
    coords = np.reshape(coords, (-1, 3))
    znucl_type, typat = (kwargs['znucl'], kwargs['typat'])
    if not isinstance(znucl_type, Iterable):
        znucl_type = [znucl_type]
    if not isinstance(typat, Iterable):
        typat = [typat]
    if len(typat) != len(coords):
        raise ValueError(f'len(typat)={len(typat)!r} must equal len(coords)={len(coords)!r}')
    typat = np.array(typat, dtype=int)
    species = [znucl_type[typ - 1] for typ in typat]
    return cls(lattice, species, coords, validate_proximity=False, to_unit_cell=False, coords_are_cartesian=coords_are_cartesian)