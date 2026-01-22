from __future__ import annotations
import copy
import itertools
import logging
import math
import typing
import warnings
from collections import Counter
from typing import TYPE_CHECKING, Literal, cast, no_type_check
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import palettable
import scipy.interpolate as scint
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from monty.dev import requires
from monty.json import jsanitize
from pymatgen.core import Element
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.boltztrap import BoltztrapError
from pymatgen.electronic_structure.core import OrbitalType, Spin
from pymatgen.util.plotting import add_fig_kwargs, get_ax3d_fig, pretty_plot
def bs_plot_data(self, zero_to_efermi=True, bs=None, bs_ref=None, split_branches=True):
    """Get the data nicely formatted for a plot.

        Args:
            zero_to_efermi: Automatically set the Fermi level as the plot's origin (i.e. subtract E - E_f).
                Defaults to True.
            bs: the bandstructure to get the data from. If not provided, the first
                one in the self._bs list will be used.
            bs_ref: is the bandstructure of reference when a rescale of the distances
                is need to plot multiple bands
            split_branches: if True distances and energies are split according to the
                branches. If False distances and energies are split only where branches
                are discontinuous (reducing the number of lines to plot).

        Returns:
            dict: A dictionary of the following format:
            ticks: A dict with the 'distances' at which there is a kpoint (the
            x axis) and the labels (None if no label).
            energy: A dict storing bands for spin up and spin down data
            {Spin:[np.array(nb_bands,kpoints),...]} as a list of discontinuous kpath
            of energies. The energy of multiple continuous branches are stored together.
            vbm: A list of tuples (distance,energy) marking the vbms. The
            energies are shifted with respect to the Fermi level is the
            option has been selected.
            cbm: A list of tuples (distance,energy) marking the cbms. The
            energies are shifted with respect to the Fermi level is the
            option has been selected.
            lattice: The reciprocal lattice.
            zero_energy: This is the energy used as zero for the plot.
            band_gap:A string indicating the band gap and its nature (empty if
            it's a metal).
            is_metal: True if the band structure is metallic (i.e., there is at
            least one band crossing the Fermi level).
        """
    if bs is None:
        bs = self._bs[0] if isinstance(self._bs, list) else self._bs
    energies = {str(sp): [] for sp in bs.bands}
    bs_is_metal = bs.is_metal()
    if not bs_is_metal:
        vbm = bs.get_vbm()
        cbm = bs.get_cbm()
    zero_energy = 0.0
    if zero_to_efermi:
        zero_energy = bs.efermi if bs_is_metal else vbm['energy']
    distances = bs.distance
    if bs_ref is not None and bs_ref.branches != bs.branches:
        distances = self._rescale_distances(bs_ref, bs)
    if split_branches:
        steps = [br['end_index'] + 1 for br in bs.branches][:-1]
    else:
        steps = self._get_branch_steps(bs.branches)[1:-1]
    distances = np.split(distances, steps)
    for sp in bs.bands:
        energies[str(sp)] = np.hsplit(bs.bands[sp] - zero_energy, steps)
    ticks = self.get_ticks()
    vbm_plot = []
    cbm_plot = []
    bg_str = ''
    if not bs_is_metal:
        for index in cbm['kpoint_index']:
            cbm_plot.append((bs.distance[index], cbm['energy'] - zero_energy if zero_to_efermi else cbm['energy']))
        for index in vbm['kpoint_index']:
            vbm_plot.append((bs.distance[index], vbm['energy'] - zero_energy if zero_to_efermi else vbm['energy']))
        bg = bs.get_band_gap()
        direct = 'Indirect'
        if bg['direct']:
            direct = 'Direct'
        bg_str = f'{direct} {bg['transition']} bandgap = {bg['energy']}'
    return {'ticks': ticks, 'distances': distances, 'energy': energies, 'vbm': vbm_plot, 'cbm': cbm_plot, 'lattice': bs.lattice_rec.as_dict(), 'zero_energy': zero_energy, 'is_metal': bs_is_metal, 'band_gap': bg_str}