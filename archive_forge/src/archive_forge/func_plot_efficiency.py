from __future__ import annotations
import collections
import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from pymatgen.io.core import ParseError
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig
@add_fig_kwargs
def plot_efficiency(self, key='wall_time', what='good+bad', nmax=5, ax: plt.Axes=None, **kwargs):
    """
        Plot the parallel efficiency.

        Args:
            key: Parallel efficiency is computed using the wall_time.
            what: Specifies what to plot: `good` for sections with good parallel efficiency.
                `bad` for sections with bad efficiency. Options can be concatenated with `+`.
            nmax: Maximum number of entries in plot
            ax: matplotlib Axes or None if a new figure should be created.

        ================  ====================================================
        kwargs            Meaning
        ================  ====================================================
        linewidth         matplotlib linewidth. Default: 2.0
        markersize        matplotlib markersize. Default: 10
        ================  ====================================================

        Returns:
            `matplotlib` figure
        """
    ax, fig = get_ax_fig(ax=ax)
    lw = kwargs.pop('linewidth', 2.0)
    msize = kwargs.pop('markersize', 10)
    what = what.split('+')
    timers = self.timers()
    peff = self.pefficiency()
    n = len(timers)
    xx = np.arange(n)
    ax.set_prop_cycle(color=['g', 'b', 'c', 'm', 'y', 'k'])
    lines, legend_entries = ([], [])
    if 'good' in what:
        good = peff.good_sections(key=key, nmax=nmax)
        for g in good:
            yy = peff[g][key]
            line, = ax.plot(xx, yy, '-->', linewidth=lw, markersize=msize)
            lines.append(line)
            legend_entries.append(g)
    if 'bad' in what:
        bad = peff.bad_sections(key=key, nmax=nmax)
        for b in bad:
            yy = peff[b][key]
            line, = ax.plot(xx, yy, '-.<', linewidth=lw, markersize=msize)
            lines.append(line)
            legend_entries.append(b)
    if 'total' not in legend_entries:
        yy = peff['total'][key]
        total_line, = ax.plot(xx, yy, 'r', linewidth=lw, markersize=msize)
        lines.append(total_line)
        legend_entries.append('total')
    ax.legend(lines, legend_entries, loc='best', shadow=True)
    ax.set_xlabel('Total_NCPUs')
    ax.set_ylabel('Efficiency')
    ax.grid(visible=True)
    labels = [f'MPI={timer.mpi_nprocs}, OMP={timer.omp_nthreads}' for timer in timers]
    ax.set_xticks(xx)
    ax.set_xticklabels(labels, fontdict=None, minor=False, rotation=15)
    return fig