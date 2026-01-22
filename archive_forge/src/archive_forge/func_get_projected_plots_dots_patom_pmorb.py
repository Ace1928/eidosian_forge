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
def get_projected_plots_dots_patom_pmorb(self, dictio, dictpa, sum_atoms=None, sum_morbs=None, zero_to_efermi=True, ylim=None, vbm_cbm_marker=False, selected_branches=None, w_h_size=(12, 8), num_column=None):
    """Method returns a plot composed of subplots for different atoms and
        orbitals (subshell orbitals such as 's', 'p', 'd' and 'f' defined by
        azimuthal quantum numbers l = 0, 1, 2 and 3, respectively or
        individual orbitals like 'px', 'py' and 'pz' defined by magnetic
        quantum numbers m = -1, 1 and 0, respectively).
        This is an extension of "get_projected_plots_dots" method.

        Args:
            dictio: The elements and the orbitals you need to project on. The
                format is {Element:[Orbitals]}, for instance:
                {'Cu':['dxy','s','px'],'O':['px','py','pz']} will give projections for Cu on
                orbitals dxy, s, px and for O on orbitals px, py, pz. If you want to sum over all
                individual orbitals of subshell orbitals, for example, 'px', 'py' and 'pz' of O,
                just simply set {'Cu':['dxy','s','px'],'O':['p']} and set sum_morbs (see
                explanations below) as {'O':[p],...}. Otherwise, you will get an error.
            dictpa: The elements and their sites (defined by site numbers) you
                need to project on. The format is {Element: [Site numbers]}, for instance:
                {'Cu':[1,5],'O':[3,4]} will give projections for Cu on site-1 and on site-5, O on
                site-3 and on site-4 in the cell. The correct site numbers of atoms are consistent
                with themselves in the structure computed. Normally, the structure should be totally
                similar with POSCAR file, however, sometimes VASP can rotate or translate the cell.
                Thus, it would be safe if using Vasprun class to get the final_structure and as a
                result, correct index numbers of atoms.
            sum_atoms: Sum projection of the similar atoms together (e.g.: Cu
                on site-1 and Cu on site-5). The format is {Element: [Site numbers]}, for instance:
                {'Cu': [1,5], 'O': [3,4]} means summing projections over Cu on site-1 and Cu on
                site-5 and O on site-3 and on site-4. If you do not want to use this functional,
                just turn it off by setting sum_atoms = None.
            sum_morbs: Sum projections of individual orbitals of similar atoms
                together (e.g.: 'dxy' and 'dxz'). The format is {Element: [individual orbitals]},
                for instance: {'Cu': ['dxy', 'dxz'], 'O': ['px', 'py']} means summing projections
                over 'dxy' and 'dxz' of Cu and 'px' and 'py' of O. If you do not want to use this
                functional, just turn it off by setting sum_morbs = None.
            zero_to_efermi: Automatically set the Fermi level as the plot's origin (i.e. subtract E - E_f).
                Defaults to True.
            ylim: The y-axis limit. Defaults to None.
            vbm_cbm_marker: Whether to plot points to indicate valence band maxima and conduction
                band minima positions. Defaults to False.
            selected_branches: The index of symmetry lines you chose for
                plotting. This can be useful when the number of symmetry lines (in KPOINTS file) are
                manny while you only want to show for certain ones. The format is [index of line],
                for instance: [1, 3, 4] means you just need to do projection along lines number 1, 3
                and 4 while neglecting lines number 2 and so on. By default, this is None type and
                all symmetry lines will be plotted.
            w_h_size: This variable help you to control the width and height
                of figure. By default, width = 12 and height = 8 (inches). The width/height ratio is
                kept the same for subfigures and the size of each depends on how many number of
                subfigures are plotted.
            num_column: This variable help you to manage how the subfigures are
                arranged in the figure by setting up the number of columns of subfigures. The value
                should be an int number. For example, num_column = 3 means you want to plot
                subfigures in 3 columns. By default, num_column = None and subfigures are aligned in
                2 columns.

        Returns:
            A pyplot object with different subfigures for different projections.
            The blue and red colors lines are bands
            for spin up and spin down. The green and cyan dots are projections
            for spin up and spin down. The bigger
            the green or cyan dots in the projected band structures, the higher
            character for the corresponding elements
            and orbitals. List of individual orbitals and their numbers (set up
            by VASP and no special meaning):
            s = 0; py = 1 pz = 2 px = 3; dxy = 4 dyz = 5 dz2 = 6 dxz = 7 dx2 = 8;
            f_3 = 9 f_2 = 10 f_1 = 11 f0 = 12 f1 = 13 f2 = 14 f3 = 15
        """
    dictio, sum_morbs = self._Orbitals_SumOrbitals(dictio, sum_morbs)
    dictpa, sum_atoms, n_figs = self._number_of_subfigures(dictio, dictpa, sum_atoms, sum_morbs)
    print(f'Number of subfigures: {n_figs}')
    if n_figs > 9:
        print(f'The number of subfigures {n_figs} might be too manny and the implementation might take a long time.\n A smaller number or a plot with selected symmetry lines (selected_branches) might be better.')
    band_linewidth = 0.5
    ax = pretty_plot(w_h_size[0], w_h_size[1])
    proj_br_d, dictio_d, dictpa_d, branches = self._get_projections_by_branches_patom_pmorb(dictio, dictpa, sum_atoms, sum_morbs, selected_branches)
    data = self.bs_plot_data(zero_to_efermi)
    e_min = -4
    e_max = 4
    if self._bs.is_metal():
        e_min = -10
        e_max = 10
    count = 0
    for elt in dictpa_d:
        for numa in dictpa_d[elt]:
            for o in dictio_d[elt]:
                count += 1
                if num_column is None:
                    if n_figs == 1:
                        plt.subplot(1, 1, 1)
                    else:
                        row = n_figs // 2
                        if n_figs % 2 == 0:
                            plt.subplot(row, 2, count)
                        else:
                            plt.subplot(row + 1, 2, count)
                elif isinstance(num_column, int):
                    row = n_figs / num_column
                    if n_figs % num_column == 0:
                        plt.subplot(row, num_column, count)
                    else:
                        plt.subplot(row + 1, num_column, count)
                else:
                    raise ValueError("The invalid 'num_column' is assigned. It should be an integer.")
                ax, shift = self._make_ticks_selected(ax, branches)
                br = -1
                for b in branches:
                    br += 1
                    for band_idx in range(self._nb_bands):
                        ax.plot([x - shift[br] for x in data['distances'][b]], [data['energy'][str(Spin.up)][b][band_idx][j] for j in range(len(data['distances'][b]))], 'b-', linewidth=band_linewidth)
                        if self._bs.is_spin_polarized:
                            ax.plot([x - shift[br] for x in data['distances'][b]], [data['energy'][str(Spin.down)][b][band_idx][j] for j in range(len(data['distances'][b]))], 'r--', linewidth=band_linewidth)
                            for j in range(len(data['energy'][str(Spin.up)][b][band_idx])):
                                ax.plot(data['distances'][b][j] - shift[br], data['energy'][str(Spin.down)][b][band_idx][j], 'co', markersize=proj_br_d[br][str(Spin.down)][band_idx][j][elt + numa][o] * 15.0)
                        for j in range(len(data['energy'][str(Spin.up)][b][band_idx])):
                            ax.plot(data['distances'][b][j] - shift[br], data['energy'][str(Spin.up)][b][band_idx][j], 'go', markersize=proj_br_d[br][str(Spin.up)][band_idx][j][elt + numa][o] * 15.0)
                if ylim is None:
                    if self._bs.is_metal():
                        if zero_to_efermi:
                            ax.set_ylim(e_min, e_max)
                        else:
                            ax.set_ylim(self._bs.efermi + e_min, self._bs._efermi + e_max)
                    else:
                        if vbm_cbm_marker:
                            for cbm in data['cbm']:
                                ax.scatter(cbm[0], cbm[1], color='r', marker='o', s=100)
                            for vbm in data['vbm']:
                                ax.scatter(vbm[0], vbm[1], color='g', marker='o', s=100)
                        ax.set_ylim(data['vbm'][0][1] + e_min, data['cbm'][0][1] + e_max)
                else:
                    ax.set_ylim(ylim)
                ax.set_title(f'{elt} {numa} {o}')
    return ax