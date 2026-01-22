from typing import Dict, Tuple
import numpy as np
from cirq import value
from cirq._doc import document
Computes non-negative integer power of single-qubit Pauli combination.

    Returns scalar coefficients bi, bx, by, bz such that

        bi I + bx X + by Y + bz Z = (ai I + ax X + ay Y + az Z)^exponent

    Correctness of the formulas below follows from the binomial expansion
    and the fact that for any real or complex vector (ax, ay, az) and any
    non-negative integer k:

         [ax X + ay Y + az Z]^(2k) = (ax^2 + ay^2 + az^2)^k I

    