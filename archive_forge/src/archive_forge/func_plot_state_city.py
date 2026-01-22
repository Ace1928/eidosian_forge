from typing import List, Union
from functools import reduce
import colorsys
import numpy as np
from qiskit import user_config
from qiskit.quantum_info.states.statevector import Statevector
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic import PauliList, SparsePauliOp
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.utils import optionals as _optionals
from qiskit.circuit.tools.pi_check import pi_check
from .array import _num_to_latex, array_to_latex
from .utils import matplotlib_close_if_inline
from .exceptions import VisualizationError
@_optionals.HAS_MATPLOTLIB.require_in_call
def plot_state_city(state, title='', figsize=None, color=None, alpha=1, ax_real=None, ax_imag=None, *, filename=None):
    """Plot the cityscape of quantum state.

    Plot two 3d bar graphs (two dimensional) of the real and imaginary
    part of the density matrix rho.

    Args:
        state (Statevector or DensityMatrix or ndarray): an N-qubit quantum state.
        title (str): a string that represents the plot title
        figsize (tuple): Figure size in inches.
        color (list): A list of len=2 giving colors for real and
            imaginary components of matrix elements.
        alpha (float): Transparency value for bars
        ax_real (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. If this is specified without an
            ax_imag only the real component plot will be generated.
            Additionally, if specified there will be no returned Figure since
            it is redundant.
        ax_imag (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. If this is specified without an
            ax_real only the imaginary component plot will be generated.
            Additionally, if specified there will be no returned Figure since
            it is redundant.

    Returns:
        :class:`matplotlib:matplotlib.figure.Figure` :
            The matplotlib.Figure of the visualization if the
            ``ax_real`` and ``ax_imag`` kwargs are not set

    Raises:
        MissingOptionalLibraryError: Requires matplotlib.
        ValueError: When 'color' is not a list of len=2.
        VisualizationError: if input is not a valid N-qubit state.

    Examples:
        .. plot::
           :include-source:

           # You can choose different colors for the real and imaginary parts of the density matrix.

           from qiskit import QuantumCircuit
           from qiskit.quantum_info import DensityMatrix
           from qiskit.visualization import plot_state_city

           qc = QuantumCircuit(2)
           qc.h(0)
           qc.cx(0, 1)

           state = DensityMatrix(qc)
           plot_state_city(state, color=['midnightblue', 'crimson'], title="New State City")

        .. plot::
           :include-source:

           # You can make the bars more transparent to better see the ones that are behind
           # if they overlap.

           import numpy as np
           from qiskit.quantum_info import Statevector
           from qiskit.visualization import plot_state_city
           from qiskit import QuantumCircuit

           qc = QuantumCircuit(2)
           qc.h(0)
           qc.cx(0, 1)


           qc = QuantumCircuit(2)
           qc.h([0, 1])
           qc.cz(0,1)
           qc.ry(np.pi/3, 0)
           qc.rx(np.pi/5, 1)

           state = Statevector(qc)
           plot_state_city(state, alpha=0.6)

    """
    import matplotlib.colors as mcolors
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    rho = DensityMatrix(state)
    num = rho.num_qubits
    if num is None:
        raise VisualizationError('Input is not a multi-qubit quantum state.')
    datareal = np.real(rho.data)
    dataimag = np.imag(rho.data)
    column_names = [bin(i)[2:].zfill(num) for i in range(2 ** num)]
    row_names = [bin(i)[2:].zfill(num) for i in range(2 ** num)]
    ly, lx = datareal.shape[:2]
    xpos = np.arange(0, lx, 1)
    ypos = np.arange(0, ly, 1)
    xpos, ypos = np.meshgrid(xpos + 0.25, ypos + 0.25)
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(lx * ly)
    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dzr = datareal.flatten()
    dzi = dataimag.flatten()
    if color is None:
        real_color, imag_color = ('#648fff', '#648fff')
    else:
        if len(color) != 2:
            raise ValueError("'color' must be a list of len=2.")
        real_color = '#648fff' if color[0] is None else color[0]
        imag_color = '#648fff' if color[1] is None else color[1]
    if ax_real is None and ax_imag is None:
        if figsize is None:
            figsize = (16, 8)
        fig = plt.figure(figsize=figsize, facecolor='w')
        ax1 = fig.add_subplot(1, 2, 1, projection='3d', computed_zorder=False)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d', computed_zorder=False)
    elif ax_real is not None:
        fig = ax_real.get_figure()
        ax1 = ax_real
        ax2 = ax_imag
    else:
        fig = ax_imag.get_figure()
        ax1 = None
        ax2 = ax_imag
    fig.tight_layout()
    max_dzr = np.max(dzr)
    max_dzi = np.max(dzi)
    fig_width, fig_height = fig.get_size_inches()
    max_plot_size = min(fig_width / 2.25, fig_height)
    max_font_size = int(3 * max_plot_size)
    max_zoom = 10 / (10 + np.sqrt(max_plot_size))
    for ax, dz, col, zlabel in ((ax1, dzr, real_color, 'Real'), (ax2, dzi, imag_color, 'Imaginary')):
        if ax is None:
            continue
        max_dz = np.max(dz)
        min_dz = np.min(dz)
        if isinstance(col, str) and col.startswith('#'):
            col = mcolors.to_rgba_array(col)
        dzn = dz < 0
        if np.any(dzn):
            fc = generate_facecolors(xpos[dzn], ypos[dzn], zpos[dzn], dx[dzn], dy[dzn], dz[dzn], col)
            negative_bars = ax.bar3d(xpos[dzn], ypos[dzn], zpos[dzn], dx[dzn], dy[dzn], dz[dzn], alpha=alpha, zorder=0.625)
            negative_bars.set_facecolor(fc)
        if min_dz < 0 < max_dz:
            xlim, ylim = ([0, lx], [0, ly])
            verts = [list(zip(xlim + xlim[::-1], np.repeat(ylim, 2), [0] * 4))]
            plane = Poly3DCollection(verts, alpha=0.25, facecolor='k', linewidths=1)
            plane.set_zorder(0.75)
            ax.add_collection3d(plane)
        dzp = dz >= 0
        if np.any(dzp):
            fc = generate_facecolors(xpos[dzp], ypos[dzp], zpos[dzp], dx[dzp], dy[dzp], dz[dzp], col)
            positive_bars = ax.bar3d(xpos[dzp], ypos[dzp], zpos[dzp], dx[dzp], dy[dzp], dz[dzp], alpha=alpha, zorder=0.875)
            positive_bars.set_facecolor(fc)
        ax.set_title(f'{zlabel} Amplitude (ρ)', fontsize=max_font_size)
        ax.set_xticks(np.arange(0.5, lx + 0.5, 1))
        ax.set_yticks(np.arange(0.5, ly + 0.5, 1))
        if max_dz != min_dz:
            ax.axes.set_zlim3d(min_dz, max(max_dzr + 1e-09, max_dzi))
        elif min_dz == 0:
            ax.axes.set_zlim3d(min_dz, max(max_dzr + 1e-09, max_dzi))
        else:
            ax.axes.set_zlim3d(auto=True)
        ax.get_autoscalez_on()
        ax.xaxis.set_ticklabels(row_names, fontsize=max_font_size, rotation=45, ha='right', va='top')
        ax.yaxis.set_ticklabels(column_names, fontsize=max_font_size, rotation=-22.5, ha='left', va='center')
        for tick in ax.zaxis.get_major_ticks():
            tick.label1.set_fontsize(max_font_size)
            tick.label1.set_horizontalalignment('left')
            tick.label1.set_verticalalignment('bottom')
        ax.set_box_aspect(aspect=(4, 4, 4), zoom=max_zoom)
        ax.set_xmargin(0)
        ax.set_ymargin(0)
    fig.suptitle(title, fontsize=max_font_size * 1.25)
    fig.subplots_adjust(top=0.9, bottom=0, left=0, right=1, hspace=0, wspace=0)
    if ax_real is None and ax_imag is None:
        matplotlib_close_if_inline(fig)
    if filename is None:
        return fig
    else:
        return fig.savefig(filename)