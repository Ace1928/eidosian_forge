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
def state_drawer(state, output=None, **drawer_args):
    """Returns a visualization of the state.

    **repr**: ASCII TextMatrix of the state's ``_repr_``.

    **text**: ASCII TextMatrix that can be printed in the console.

    **latex**: An IPython Latex object for displaying in Jupyter Notebooks.

    **latex_source**: Raw, uncompiled ASCII source to generate array using LaTeX.

    **qsphere**: Matplotlib figure, rendering of statevector using `plot_state_qsphere()`.

    **hinton**: Matplotlib figure, rendering of statevector using `plot_state_hinton()`.

    **bloch**: Matplotlib figure, rendering of statevector using `plot_bloch_multivector()`.

    **city**: Matplotlib figure, rendering of statevector using `plot_state_city()`.

    **paulivec**: Matplotlib figure, rendering of statevector using `plot_state_paulivec()`.

    Args:
        output (str): Select the output method to use for drawing the
            circuit. Valid choices are ``text``, ``latex``, ``latex_source``,
            ``qsphere``, ``hinton``, ``bloch``, ``city`` or ``paulivec``.
            Default is `'text`'.
        drawer_args: Arguments to be passed to the relevant drawer. For
            'latex' and 'latex_source' see ``array_to_latex``

    Returns:
        :class:`matplotlib.figure` or :class:`str` or
        :class:`TextMatrix` or :class:`IPython.display.Latex`:
        Drawing of the state.

    Raises:
        MissingOptionalLibraryError: when `output` is `latex` and IPython is not installed.
        ValueError: when `output` is not a valid selection.
    """
    config = user_config.get_config()
    default_output = 'repr'
    if output is None:
        if config:
            default_output = config.get('state_drawer', 'repr')
        output = default_output
    output = output.lower()
    drawers = {'text': TextMatrix, 'latex_source': state_to_latex, 'qsphere': plot_state_qsphere, 'hinton': plot_state_hinton, 'bloch': plot_bloch_multivector, 'city': plot_state_city, 'paulivec': plot_state_paulivec}
    if output == 'latex':
        _optionals.HAS_IPYTHON.require_now('state_drawer')
        from IPython.display import Latex
        draw_func = drawers['latex_source']
        return Latex(f'$${draw_func(state, **drawer_args)}$$')
    if output == 'repr':
        return state.__repr__()
    try:
        draw_func = drawers[output]
        return draw_func(state, **drawer_args)
    except KeyError as err:
        raise ValueError("'{}' is not a valid option for drawing {} objects. Please choose from:\n            'text', 'latex', 'latex_source', 'qsphere', 'hinton',\n            'bloch', 'city' or 'paulivec'.".format(output, type(state).__name__)) from err