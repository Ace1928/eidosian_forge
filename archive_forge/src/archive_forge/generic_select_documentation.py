from typing import Collection, Optional, Sequence, Tuple, Union
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import select_and_prepare, unary_iteration_gate
Applies `self.select_unitaries[selection]`.

        Args:
             context: `cirq.DecompositionContext` stores options for decomposing gates (eg:
                cirq.QubitManager).
             selection: takes on values [0, self.iteration_lengths[0])
             control: Qid that is the control qubit or qubits
             target: Target register qubits
        