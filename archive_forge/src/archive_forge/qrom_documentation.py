from typing import Callable, Sequence, Tuple
import attr
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import and_gate, unary_iteration_gate
from numpy.typing import ArrayLike, NDArray
Gate to load data[l] in the target register when the selection stores an index l.

    In the case of multi-dimensional data[p,q,r,...] we use multiple named
    selection signature [p, q, r, ...] to index and load the data. Here `p, q, r, ...`
    correspond to signature named `selection0`, `selection1`, `selection2`, ... etc.

    When the input data elements contain consecutive entries of identical data elements to
    load, the QROM also implements the "variable-spaced" QROM optimization described in Ref[2].

    Args:
        data: List of numpy ndarrays specifying the data to load. If the length
            of this list is greater than one then we use the same selection indices
            to load each dataset (for example, to load alt and keep data for
            state preparation). Each data set is required to have the same
            shape and to be of integer type.
        selection_bitsizes: The number of bits used to represent each selection register
            corresponding to the size of each dimension of the array. Should be
            the same length as the shape of each of the datasets.
        target_bitsizes: The number of bits used to represent the data
            signature. This can be deduced from the maximum element of each of the
            datasets. Should be of length len(data), i.e. the number of datasets.
        num_controls: The number of control signature.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity]
        (https://arxiv.org/abs/1805.03662).
            Babbush et. al. (2018). Figure 1.

        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization]
        (https://arxiv.org/abs/2007.07391).
            Babbush et. al. (2020). Figure 3.
    