from typing import Any, Dict, List, Sequence, Union
import numpy as np
import cirq
from cirq import protocols, qis, value
from cirq.value import big_endian_int_to_digits, random_state
def to_state_vector(self) -> np.ndarray:
    arr = np.zeros(2 ** self.n, dtype=complex)
    for x in range(len(arr)):
        arr[x] = self.inner_product_of_state_and_x(x)
    return arr