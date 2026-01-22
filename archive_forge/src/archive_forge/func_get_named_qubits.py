import enum
import abc
import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, overload, Iterator
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq_ft.deprecation import deprecated_cirq_ft_class
def get_named_qubits(registers: Iterable[Register]) -> Dict[str, NDArray[cirq.Qid]]:
    """Returns a dictionary of appropriately shaped named qubit signature for input `signature`."""

    def _qubit_array(reg: Register):
        qubits = np.empty(reg.shape + (reg.bitsize,), dtype=object)
        for ii in reg.all_idxs():
            for j in range(reg.bitsize):
                prefix = '' if not ii else f'[{', '.join((str(i) for i in ii))}]'
                suffix = '' if reg.bitsize == 1 else f'[{j}]'
                qubits[ii + (j,)] = cirq.NamedQubit(reg.name + prefix + suffix)
        return qubits

    def _qubits_for_reg(reg: Register):
        if len(reg.shape) > 0:
            return _qubit_array(reg)
        return np.array([cirq.NamedQubit(f'{reg.name}')] if reg.total_bits() == 1 else cirq.NamedQubit.range(reg.total_bits(), prefix=reg.name), dtype=object)
    return {reg.name: _qubits_for_reg(reg) for reg in registers}