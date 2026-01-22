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
@requires(mlab is not None, 'MayAvi mlab not imported! Please install mayavi.')
def plot_fermi_surface(data, structure, cbm, energy_levels=None, multiple_figure=True, mlab_figure=None, kpoints_dict=None, colors=None, transparency_factor=None, labels_scale_factor=0.05, points_scale_factor=0.02, interactive=True):
    """Plot the Fermi surface at specific energy value using Boltztrap 1 FERMI
    mode.

    The easiest way to use this plotter is:

        1. Run boltztrap in 'FERMI' mode using BoltztrapRunner,
        2. Load BoltztrapAnalyzer using your method of choice (e.g., from_files)
        3. Pass in your BoltztrapAnalyzer's fermi_surface_data as this
            function's data argument.

    Args:
        data: energy values in a 3D grid from a CUBE file via read_cube_file
            function, or from a BoltztrapAnalyzer.fermi_surface_data
        structure: structure object of the material
        energy_levels ([float]): Energy values for plotting the fermi surface(s)
            By default 0 eV correspond to the VBM, as in the plot of band
            structure along symmetry line.
            Default: One surface, with max energy value + 0.01 eV
        cbm (bool): Boolean value to specify if the considered band is a
            conduction band or not
        multiple_figure (bool): If True a figure for each energy level will be
            shown. If False all the surfaces will be shown in the same figure.
            In this last case, tune the transparency factor.
        mlab_figure (mayavi.mlab.figure): A previous figure to plot a new
            surface on.
        kpoints_dict (dict): dictionary of kpoints to label in the plot.
            Example: {"K":[0.5,0.0,0.5]}, coords are fractional
        colors ([tuple]): Iterable of 3-tuples (r,g,b) of integers to define
            the colors of each surface (one per energy level).
            Should be the same length as the number of surfaces being plotted.
            Example (3 surfaces): colors=[(1,0,0), (0,1,0), (0,0,1)]
            Example (2 surfaces): colors=[(0, 0.5, 0.5)]
        transparency_factor (float): Values in the range [0,1] to tune the
            opacity of each surface. Should be one transparency_factor per
            surface.
        labels_scale_factor (float): factor to tune size of the kpoint labels
        points_scale_factor (float): factor to tune size of the kpoint points
        interactive (bool): if True an interactive figure will be shown.
            If False a non interactive figure will be shown, but it is possible
            to plot other surfaces on the same figure. To make it interactive,
            run mlab.show().

    Returns:
        tuple[mlab.figure, mlab]: The mlab plotter and an interactive
            figure to control the plot.

    Note: Experimental.
        Please, double check the surface shown by using some other software and report issues.
    """
    bz = structure.lattice.reciprocal_lattice.get_wigner_seitz_cell()
    cell = structure.lattice.reciprocal_lattice.matrix
    fact = 1 if not cbm else -1
    data_1d = data.ravel()
    en_min = np.min(fact * data_1d)
    en_max = np.max(fact * data_1d)
    if energy_levels is None:
        energy_levels = [en_min + 0.01] if cbm else [en_max - 0.01]
        print(f'Energy level set to: {energy_levels[0]} eV')
    else:
        for e in energy_levels:
            if e > en_max or e < en_min:
                raise BoltztrapError(f'energy level {e} not in the range of possible energies: [{en_min}, {en_max}]')
    n_surfaces = len(energy_levels)
    if colors is None:
        colors = [(0, 0, 1)] * n_surfaces
    if transparency_factor is None:
        transparency_factor = [1] * n_surfaces
    if mlab_figure:
        fig = mlab_figure
    if kpoints_dict is None:
        kpoints_dict = {}
    if mlab_figure is None and (not multiple_figure):
        fig = mlab.figure(size=(1024, 768), bgcolor=(1, 1, 1))
        for iface in range(len(bz)):
            for line in itertools.combinations(bz[iface], 2):
                for jface in range(len(bz)):
                    if iface < jface and any((np.all(line[0] == x) for x in bz[jface])) and any((np.all(line[1] == x) for x in bz[jface])):
                        mlab.plot3d(*zip(line[0], line[1]), color=(0, 0, 0), tube_radius=None, figure=fig)
        for key, coords in kpoints_dict.items():
            label_coords = structure.lattice.reciprocal_lattice.get_cartesian_coords(coords)
            mlab.points3d(*label_coords, scale_factor=points_scale_factor, color=(0, 0, 0), figure=fig)
            mlab.text3d(*label_coords, text=key, scale=labels_scale_factor, color=(0, 0, 0), figure=fig)
    for i, isolevel in enumerate(energy_levels):
        alpha = transparency_factor[i]
        color = colors[i]
        if multiple_figure:
            fig = mlab.figure(size=(1024, 768), bgcolor=(1, 1, 1))
            for iface in range(len(bz)):
                for line in itertools.combinations(bz[iface], 2):
                    for jface in range(len(bz)):
                        if iface < jface and any((np.all(line[0] == x) for x in bz[jface])) and any((np.all(line[1] == x) for x in bz[jface])):
                            mlab.plot3d(*zip(line[0], line[1]), color=(0, 0, 0), tube_radius=None, figure=fig)
            for key, coords in kpoints_dict.items():
                label_coords = structure.lattice.reciprocal_lattice.get_cartesian_coords(coords)
                mlab.points3d(*label_coords, scale_factor=points_scale_factor, color=(0, 0, 0), figure=fig)
                mlab.text3d(*label_coords, text=key, scale=labels_scale_factor, color=(0, 0, 0), figure=fig)
        cp = mlab.contour3d(fact * data, contours=[isolevel], transparent=True, colormap='hot', color=color, opacity=alpha, figure=fig)
        polydata = cp.actor.actors[0].mapper.input
        pts = np.array(polydata.points)
        polydata.points = np.dot(pts, cell / np.array(data.shape)[:, np.newaxis])
        cx, cy, cz = (np.mean(np.array(polydata.points)[:, i]) for i in range(3))
        polydata.points = (np.array(polydata.points) - [cx, cy, cz]) * 2
        fig.scene.isometric_view()
    if interactive:
        mlab.show()
    return (fig, mlab)