import random
import numpy as np
import pytest
import sympy
import cirq

    # Canonicalize X exponent (-1, +1].
    if isinstance(x, numbers.Real):
        x %= 2
        if x > 1:
            x -= 2
    # Axis phase exponent is irrelevant if there is no X exponent.
    # Canonicalize Z exponent (-1, +1].
    if isinstance(z, numbers.Real):
        z %= 2
        if z > 1:
            z -= 2

    # Canonicalize axis phase exponent into (-0.5, +0.5].
    if isinstance(a, numbers.Real):
        a %= 2
        if a > 1:
            a -= 2
        if a <= -0.5:
            a += 1
            x = -x
        elif a > 0.5:
            a -= 1
            x = -x
    