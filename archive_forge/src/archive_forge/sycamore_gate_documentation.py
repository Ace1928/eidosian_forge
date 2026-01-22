import numpy as np
import cirq
from cirq._doc import document
The Sycamore gate is a two-qubit gate equivalent to FSimGate(π/2, π/6).

    The unitary of this gate is

        [[1, 0, 0, 0],
         [0, 0, -1j, 0],
         [0, -1j, 0, 0],
         [0, 0, 0, exp(- 1j * π/6)]]

    This gate can be performed on the Google's Sycamore chip and
    is close to the gates that were used to demonstrate beyond
    classical resuts used in this paper:
    https://www.nature.com/articles/s41586-019-1666-5
    