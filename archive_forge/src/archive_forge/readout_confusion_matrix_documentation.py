import time
from typing import Any, Dict, Union, Sequence, List, Tuple, TYPE_CHECKING, Optional, cast
import sympy
import numpy as np
import scipy.optimize
from cirq import circuits, ops, vis, study
from cirq._compat import proper_repr
Applies corrections to the observed `result` to compensate for readout error on qubits.

        The compensation can applied by the following methods:
         1. 'pseudo_inverse': The result is multiplied by the correction matrix, which is pseudo
                              inverse of confusion matrix corresponding to the subspace defined by
                              `qubits`.
         2. 'least_squares': Solves a constrained minimization problem to find optimal `x` s.t.
                                a) x >= 0
                                b) sum(x) == sum(result) and
                                c) sum((result - x @ confusion_matrix) ** 2) is minimized.

        Args:
            result: `(2 ** len(qubits), )` shaped numpy array containing observed frequencies /
                    probabilities.
            qubits: Sequence of qubits used for sampling to get `result`. By default, uses all
                    qubits in sorted order, i.e. `self.qubits`. Note that ordering of qubits sets
                    the basis ordering for the `result` argument.
            method: Correction Method. Should be either 'pseudo_inverse' or 'least_squares'.
                    Equal to `least_squares` by default.

        Returns:
              `(2 ** len(qubits), )` shaped numpy array corresponding to `result` with corrections.

        Raises:
            ValueError: If `result.shape` != `(2 ** len(qubits),)`.
            ValueError: If `least_squares` constrained minimization problem does not converge.
        