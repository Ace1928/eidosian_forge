import numpy as np
import matplotlib.pyplot as plt
import mpmath

Convergence regions of the expansions used in ``struve.c``

Note that for v >> z both functions tend rapidly to 0,
and for v << -z, they tend to infinity.

The floating-point functions over/underflow in the lower left and right
corners of the figure.


Figure legend
=============

Red region
    Power series is close (1e-12) to the mpmath result

Blue region
    Asymptotic series is close to the mpmath result

Green region
    Bessel series is close to the mpmath result

Dotted colored lines
    Boundaries of the regions

Solid colored lines
    Boundaries estimated by the routine itself. These will be used
    for determining which of the results to use.

Black dashed line
    The line z = 0.7*|v| + 12

