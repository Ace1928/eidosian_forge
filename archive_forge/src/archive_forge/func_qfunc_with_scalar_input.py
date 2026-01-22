import pytest
import pennylane.numpy as pnp
import pennylane as qml
def qfunc_with_scalar_input(model=None):
    """Model dependent quantum function taking a single input"""

    def qfunc(x):
        if model == 'qubit':
            qml.RX(x, wires=0)
        elif model == 'cv':
            qml.Displacement(x, 0.0, wires=0)
        return qml.expval(qml.Identity(wires=0))
    return qfunc