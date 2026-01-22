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
def ebands_input(structure, pseudos, kppa=None, nscf_nband=None, ndivsm=15, ecut=None, pawecutdg=None, scf_nband=None, accuracy='normal', spin_mode='polarized', smearing='fermi_dirac:0.1 eV', charge=0.0, scf_algorithm=None, dos_kppa=None):
    """
    Returns a |BasicMultiDataset| object for band structure calculations.

    Args:
        structure: |Structure| object.
        pseudos: List of filenames or list of |Pseudo| objects or |PseudoTable| object.
        kppa: Defines the sampling used for the SCF run. Defaults to 1000 if not given.
        nscf_nband: Number of bands included in the NSCF run. Set to scf_nband + 10 if None.
        ndivsm: Number of divisions used to sample the smallest segment of the k-path.
                if 0, only the GS input is returned in multi[0].
        ecut: cutoff energy in Ha (if None, ecut is initialized from the pseudos according to accuracy)
        pawecutdg: cutoff energy in Ha for PAW double-grid (if None, pawecutdg is initialized from the pseudos
            according to accuracy)
        scf_nband: Number of bands for SCF run. If scf_nband is None, nband is automatically initialized
            from the list of pseudos, the structure and the smearing option.
        accuracy: Accuracy of the calculation.
        spin_mode: Spin polarization.
        smearing: Smearing technique.
        charge: Electronic charge added to the unit cell.
        scf_algorithm: Algorithm used for solving of the SCF cycle.
        dos_kppa: Scalar or List of integers with the number of k-points per atom
            to be used for the computation of the DOS (None if DOS is not wanted).
    """
    structure = as_structure(structure)
    if dos_kppa is not None and (not isinstance(dos_kppa, (list, tuple))):
        dos_kppa = [dos_kppa]
    multi = BasicMultiDataset(structure, pseudos, ndtset=2 if dos_kppa is None else 2 + len(dos_kppa))
    multi.set_vars(_find_ecut_pawecutdg(ecut, pawecutdg, multi.pseudos, accuracy))
    kppa = _DEFAULTS.get('kppa') if kppa is None else kppa
    scf_ksampling = aobj.KSampling.automatic_density(structure, kppa, chksymbreak=0)
    scf_electrons = aobj.Electrons(spin_mode=spin_mode, smearing=smearing, algorithm=scf_algorithm, charge=charge, nband=scf_nband, fband=None)
    if scf_electrons.nband is None:
        scf_electrons.nband = _find_scf_nband(structure, multi.pseudos, scf_electrons, multi[0].get('spinat'))
    multi[0].set_vars(scf_ksampling.to_abivars())
    multi[0].set_vars(scf_electrons.to_abivars())
    multi[0].set_vars(_stopping_criterion('scf', accuracy))
    if ndivsm == 0:
        return multi
    nscf_ksampling = aobj.KSampling.path_from_structure(ndivsm, structure)
    nscf_nband = scf_electrons.nband + 10 if nscf_nband is None else nscf_nband
    nscf_electrons = aobj.Electrons(spin_mode=spin_mode, smearing=smearing, algorithm={'iscf': -2}, charge=charge, nband=nscf_nband, fband=None)
    multi[1].set_vars(nscf_ksampling.to_abivars())
    multi[1].set_vars(nscf_electrons.to_abivars())
    multi[1].set_vars(_stopping_criterion('nscf', accuracy))
    if dos_kppa is not None:
        for i, kppa_ in enumerate(dos_kppa):
            dos_ksampling = aobj.KSampling.automatic_density(structure, kppa_, chksymbreak=0)
            dos_electrons = aobj.Electrons(spin_mode=spin_mode, smearing=smearing, algorithm={'iscf': -2}, charge=charge, nband=nscf_nband)
            dt = 2 + i
            multi[dt].set_vars(dos_ksampling.to_abivars())
            multi[dt].set_vars(dos_electrons.to_abivars())
            multi[dt].set_vars(_stopping_criterion('nscf', accuracy))
    return multi