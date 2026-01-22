import os
import re
from shutil import copyfile
import numpy as np
def excitations_to_wires(singles, doubles, wires=None):
    """Map the indices representing the single and double excitations
    generated with the function :func:`~.excitations` to the wires that
    the Unitary Coupled-Cluster (UCCSD) template will act on.

    Args:
        singles (list[list[int]]): list with the indices ``r``, ``p`` of the two qubits
            representing the single excitation
            :math:`\\vert r, p \\rangle = \\hat{c}_p^\\dagger \\hat{c}_r \\vert \\mathrm{HF}\\rangle`
        doubles (list[list[int]]): list with the indices ``s``, ``r``, ``q``, ``p`` of the four
            qubits representing the double excitation
            :math:`\\vert s, r, q, p \\rangle = \\hat{c}_p^\\dagger \\hat{c}_q^\\dagger
            \\hat{c}_r \\hat{c}_s \\vert \\mathrm{HF}\\rangle`
        wires (Iterable[Any]): Wires of the quantum device. If None, will use consecutive wires.

    The indices :math:`r, s` and :math:`p, q` in these lists correspond, respectively, to the
    occupied and virtual orbitals involved in the generated single and double excitations.

    Returns:
        tuple[list[list[Any]], list[list[list[Any]]]]: lists with the sequence of wires,
        resulting from the single and double excitations, that the Unitary Coupled-Cluster
        (UCCSD) template will act on.

    **Example**

    >>> singles = [[0, 2], [1, 3]]
    >>> doubles = [[0, 1, 2, 3]]
    >>> singles_wires, doubles_wires = excitations_to_wires(singles, doubles)
    >>> print(singles_wires)
    [[0, 1, 2], [1, 2, 3]]
    >>> print(doubles_wires)
    [[[0, 1], [2, 3]]]

    >>> wires=['a0', 'b1', 'c2', 'd3']
    >>> singles_wires, doubles_wires = excitations_to_wires(singles, doubles, wires=wires)
    >>> print(singles_wires)
    [['a0', 'b1', 'c2'], ['b1', 'c2', 'd3']]
    >>> print(doubles_wires)
    [[['a0', 'b1'], ['c2', 'd3']]]
    """
    if not singles and (not doubles):
        raise ValueError(f"'singles' and 'doubles' lists can not be both empty; got singles = {singles}, doubles = {doubles}")
    expected_shape = (2,)
    for single_ in singles:
        if np.array(single_).shape != expected_shape:
            raise ValueError(f"Expected entries of 'singles' to be of shape (2,); got {np.array(single_).shape}")
    expected_shape = (4,)
    for double_ in doubles:
        if np.array(double_).shape != expected_shape:
            raise ValueError(f"Expected entries of 'doubles' to be of shape (4,); got {np.array(double_).shape}")
    max_idx = 0
    if singles:
        max_idx = np.max(singles)
    if doubles:
        max_idx = max(np.max(doubles), max_idx)
    if wires is None:
        wires = range(max_idx + 1)
    elif len(wires) != max_idx + 1:
        raise ValueError(f'Expected number of wires is {max_idx + 1}; got {len(wires)}')
    singles_wires = []
    for r, p in singles:
        s_wires = [wires[i] for i in range(r, p + 1)]
        singles_wires.append(s_wires)
    doubles_wires = []
    for s, r, q, p in doubles:
        d1_wires = [wires[i] for i in range(s, r + 1)]
        d2_wires = [wires[i] for i in range(q, p + 1)]
        doubles_wires.append([d1_wires, d2_wires])
    return (singles_wires, doubles_wires)