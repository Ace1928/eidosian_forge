from typing import Any
import numpy as np
def kernel_dot(self, pA: np.ndarray, pB: np.ndarray, kernel: str) -> np.ndarray:
    k = kernel.lower()
    if k == 'poly':
        s = np.dot(pA, pB)
        s = s * self.gamma_ + self.coef0_
        return s ** self.degree_
    if k == 'sigmoid':
        s = np.dot(pA, pB)
        s = s * self.gamma_ + self.coef0_
        return np.tanh(s)
    if k == 'rbf':
        diff = pA - pB
        s = (diff * diff).sum()
        return np.exp(-self.gamma_ * s)
    if k == 'linear':
        return np.dot(pA, pB)
    raise ValueError(f'Unexpected kernel={kernel!r}.')