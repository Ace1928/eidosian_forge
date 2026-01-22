import enum
from typing import Any, List, Optional, TYPE_CHECKING, Union
import pandas as pd
import sympy
from matplotlib import pyplot as plt
from cirq import circuits, ops, study, value
from cirq._compat import proper_repr
def plot_bloch_vector(self, ax: Optional[plt.Axes]=None, **plot_kwargs: Any) -> plt.Axes:
    """Plots the estimated length of the Bloch vector versus time.

        This plot estimates the Bloch Vector by squaring the Pauli expectation
        value of X and adding it to the square of the Pauli expectation value of
        Y.  This essentially projects the state into the XY plane.

        Note that Z expectation is not considered, since T1 related amplitude
        damping will generally push this value towards |0>
        (expectation <Z> = -1) which will significantly distort the T2 numbers.

        Args:
            ax: the plt.Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.

        Returns:
            The plt.Axes containing the plot.
        """
    show_plot = not ax
    if show_plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    assert ax is not None
    ax.set_ylim(ymin=0, ymax=1)
    bloch_vector = self._expectation_pauli_x ** 2 + self._expectation_pauli_y ** 2
    ax.plot(self._expectation_pauli_x['delay_ns'], bloch_vector['value'], 'r+-', **plot_kwargs)
    ax.set_xlabel('Delay between initialization and measurement (nanoseconds)')
    ax.set_ylabel('Bloch Vector X-Y Projection Squared')
    ax.set_title('T2 Decay Experiment Data')
    if show_plot:
        fig.show()
    return ax