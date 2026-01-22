from __future__ import annotations
import abc
import copy
import json
import logging
import os
from collections import namedtuple
from collections.abc import Mapping, MutableMapping, Sequence
from enum import Enum, unique
from typing import TYPE_CHECKING
import numpy as np
from monty.collections import AttrDict
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.io.abinit import abiobjects as aobj
from pymatgen.io.abinit.pseudos import Pseudo, PseudoTable
from pymatgen.io.abinit.variable import InputVariable
from pymatgen.symmetry.bandstructure import HighSymmKpath
def ion_ioncell_relax_input(structure, pseudos, kppa=None, nband=None, ecut=None, pawecutdg=None, accuracy='normal', spin_mode='polarized', smearing='fermi_dirac:0.1 eV', charge=0.0, scf_algorithm=None, shift_mode='Monkhorst-pack'):
    """
    Returns a |BasicMultiDataset| for a structural relaxation. The first dataset optimizes the
    atomic positions at fixed unit cell. The second datasets optimizes both ions and unit cell parameters.

    Args:
        structure: |Structure| object.
        pseudos: List of filenames or list of |Pseudo| objects or |PseudoTable| object.
        kppa: Defines the sampling used for the Brillouin zone.
        nband: Number of bands included in the SCF run.
        accuracy: Accuracy of the calculation.
        spin_mode: Spin polarization.
        smearing: Smearing technique.
        charge: Electronic charge added to the unit cell.
        scf_algorithm: Algorithm used for the solution of the SCF cycle.
    """
    structure = as_structure(structure)
    multi = BasicMultiDataset(structure, pseudos, ndtset=2)
    multi.set_vars(_find_ecut_pawecutdg(ecut, pawecutdg, multi.pseudos, accuracy))
    kppa = _DEFAULTS.get('kppa') if kppa is None else kppa
    shift_mode = ShiftMode.from_object(shift_mode)
    shifts = _get_shifts(shift_mode, structure)
    ksampling = aobj.KSampling.automatic_density(structure, kppa, chksymbreak=0, shifts=shifts)
    electrons = aobj.Electrons(spin_mode=spin_mode, smearing=smearing, algorithm=scf_algorithm, charge=charge, nband=nband, fband=None)
    if electrons.nband is None:
        electrons.nband = _find_scf_nband(structure, multi.pseudos, electrons, multi[0].get('spinat'))
    ion_relax = aobj.RelaxationMethod.atoms_only(atoms_constraints=None)
    ioncell_relax = aobj.RelaxationMethod.atoms_and_cell(atoms_constraints=None)
    multi.set_vars(electrons.to_abivars())
    multi.set_vars(ksampling.to_abivars())
    multi[0].set_vars(ion_relax.to_abivars())
    multi[0].set_vars(_stopping_criterion('relax', accuracy))
    multi[1].set_vars(ioncell_relax.to_abivars())
    multi[1].set_vars(_stopping_criterion('relax', accuracy))
    return multi