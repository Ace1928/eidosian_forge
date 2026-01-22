from functools import partial
import numpy as np
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes.synthesis.plugin import UnitarySynthesisPlugin
@property
def supports_basis_gates(self):
    """The plugin does not support basis gates and by default it synthesizes a circuit using
        ``["rx", "ry", "rz", "cx"]`` gate basis."""
    return False