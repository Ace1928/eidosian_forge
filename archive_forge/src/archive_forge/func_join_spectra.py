from itertools import combinations
import numpy as np
import pennylane as qml
def join_spectra(spec1, spec2):
    """Join two sets of frequencies that belong to the same input.

    Since :math:`\\exp(i a x)\\exp(i b x) = \\exp(i (a+b) x)`, the spectra of two gates
    encoding the same :math:`x` are joined by computing the set of sums and absolute
    values of differences of their elements.
    We only compute non-negative frequencies in this subroutine and assume the inputs
    to be non-negative frequencies as well.

    Args:
        spec1 (set[float]): first spectrum
        spec2 (set[float]): second spectrum
    Returns:
        set[float]: joined spectrum
    """
    if spec1 == {0}:
        return spec2
    if spec2 == {0}:
        return spec1
    sums = set()
    diffs = set()
    for s1 in spec1:
        for s2 in spec2:
            sums.add(s1 + s2)
            diffs.add(np.abs(s1 - s2))
    return sums.union(diffs)