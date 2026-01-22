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
def plot_bloch_multivector(state, title='', figsize=None, *, reverse_bits=False, filename=None, font_size=None, title_font_size=None, title_pad=1):
    """Plot a Bloch sphere for each qubit.

    Each component :math:`(x,y,z)` of the Bloch sphere labeled as 'qubit i' represents the expected
    value of the corresponding Pauli operator acting only on that qubit, that is, the expected value
    of :math:`I_{N-1} \\otimes\\dotsb\\otimes I_{i+1}\\otimes P_i \\otimes I_{i-1}\\otimes\\dotsb\\otimes
    I_0`, where :math:`N` is the number of qubits, :math:`P\\in \\{X,Y,Z\\}` and :math:`I` is the
    identity operator.

    Args:
        state (Statevector or DensityMatrix or ndarray): an N-qubit quantum state.
        title (str): a string that represents the plot title
        figsize (tuple): size of each individual Bloch sphere figure, in inches.
        reverse_bits (bool): If True, plots qubits following Qiskit's convention [Default:False].
        font_size (float): Font size for the Bloch ball figures.
        title_font_size (float): Font size for the title.
        title_pad (float): Padding for the title (suptitle `y` position is `y=1+title_pad/100`).

    Returns:
        :class:`matplotlib:matplotlib.figure.Figure` :
            A matplotlib figure instance.

    Raises:
        MissingOptionalLibraryError: Requires matplotlib.
        VisualizationError: if input is not a valid N-qubit state.

    Examples:
        .. plot::
           :include-source:

            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            from qiskit.visualization import plot_bloch_multivector

            qc = QuantumCircuit(2)
            qc.h(0)
            qc.x(1)

            state = Statevector(qc)
            plot_bloch_multivector(state)

        .. plot::
           :include-source:

           from qiskit import QuantumCircuit
           from qiskit.quantum_info import Statevector
           from qiskit.visualization import plot_bloch_multivector

           qc = QuantumCircuit(2)
           qc.h(0)
           qc.x(1)

           # You can reverse the order of the qubits.

           from qiskit.quantum_info import DensityMatrix

           qc = QuantumCircuit(2)
           qc.h([0, 1])
           qc.t(1)
           qc.s(0)
           qc.cx(0,1)

           matrix = DensityMatrix(qc)
           plot_bloch_multivector(matrix, title='My Bloch Spheres', reverse_bits=True)

    """
    from matplotlib import pyplot as plt
    bloch_data = _bloch_multivector_data(state)[::-1] if reverse_bits else _bloch_multivector_data(state)
    num = len(bloch_data)
    if figsize is not None:
        width, height = figsize
        width *= num
    else:
        width, height = plt.figaspect(1 / num)
    default_title_font_size = font_size if font_size is not None else 16
    title_font_size = title_font_size if title_font_size is not None else default_title_font_size
    fig = plt.figure(figsize=(width, height))
    for i in range(num):
        pos = num - 1 - i if reverse_bits else i
        ax = fig.add_subplot(1, num, i + 1, projection='3d')
        plot_bloch_vector(bloch_data[i], 'qubit ' + str(pos), ax=ax, figsize=figsize, font_size=font_size)
    fig.suptitle(title, fontsize=title_font_size, y=1.0 + title_pad / 100)
    matplotlib_close_if_inline(fig)
    if filename is None:
        return fig
    else:
        return fig.savefig(filename)