from typing import (
import abc
import collections
import itertools
import sympy
from cirq import protocols
from cirq._doc import document
from cirq.study import resolver
def dict_to_zip_sweep(factor_dict: ProductOrZipSweepLike) -> Zip:
    """Zip product of sweeps from a dictionary.

    Each entry in the dictionary specifies a sweep as a mapping from the
    parameter to a value or sequence of values. The zip product of these
    sweeps is returned.

    Args:
        factor_dict: The dictionary containing the sweeps.

    Returns:
        Zip product of the sweeps.
    """
    return Zip(*(Points(k, cast(float, v) if isinstance(v, Sequence) else [v]) for k, v in factor_dict.items()))