from __future__ import annotations
import os
import re
import shutil
import subprocess
from string import Template
from typing import TYPE_CHECKING
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Molecule
@property
def infos_on_system(self):
    """Returns infos on initial parameters as in the log file of Fiesta."""
    lst = ['=========================================', 'Reading infos on system:', '', f' Number of atoms = {self._mol.composition.num_atoms} ; number of species = {len(self._mol.symbol_set)}', f' Number of valence bands = {int(self._mol.nelectrons / 2)}', f' Sigma grid specs: n_grid = {self.correlation_grid['n_grid']} ;  dE_grid = {self.correlation_grid['dE_grid']} (eV)']
    if int(self.Exc_DFT_option['rdVxcpsi']) == 1:
        lst.append(' Exchange and correlation energy read from Vxcpsi.mat')
    elif int(self.Exc_DFT_option['rdVxcpsi']) == 0:
        lst.append(' Exchange and correlation energy re-computed')
    if self.cohsex_options['eigMethod'] == 'C':
        lst += [f' Correcting  {self.cohsex_options['nv_cohsex']} valence bands and  {self.cohsex_options['nc_cohsex']} conduction bands at COHSEX level', f' Performing   {self.cohsex_options['nit_cohsex']} diagonal COHSEX iterations']
    elif self.cohsex_options['eigMethod'] == 'HF':
        lst += [f' Correcting  {self.cohsex_options['nv_cohsex']} valence bands and  {self.cohsex_options['nc_cohsex']} conduction bands at HF level', f' Performing   {self.cohsex_options['nit_cohsex']} diagonal HF iterations']
    lst += [f' Using resolution of identity : {self.cohsex_options['resMethod']}', f' Correcting  {self.GW_options['nv_corr']} valence bands and {self.GW_options['nc_corr']} conduction bands at GW level', f' Performing   {self.GW_options['nit_gw']} GW iterations']
    if int(self.bse_tddft_options['do_bse']) == 1:
        lst.append(' Dumping data for BSE treatment')
    if int(self.bse_tddft_options['do_tddft']) == 1:
        lst.append(' Dumping data for TD-DFT treatment')
    lst.extend(('', ' Atoms in cell cartesian A:'))
    symbols = list(self._mol.symbol_set)
    for site in self._mol:
        lst.append(f' {site.x} {site.y} {site.z} {int(symbols.index(site.specie.symbol)) + 1}')
    lst.append('=========================================')
    return str(lst)