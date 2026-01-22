import warnings
from collections.abc import Iterable
from string import ascii_letters as ABC
import numpy as np
import pennylane as qml
def local_snapshots(self, wires=None, snapshots=None):
    """Compute the T x n x 2 x 2 local snapshots

        For each qubit and each snapshot, compute :math:`3 U_i^\\dagger |b_i \\rangle \\langle b_i| U_i - 1`

        Args:
            wires (Iterable[int]): The wires over which to compute the snapshots. For ``wires=None`` (default) all ``n`` qubits are used.
            snapshots (Iterable[int] or int): Only compute a subset of local snapshots. For ``snapshots=None`` (default), all local snapshots are taken.
                In case of an integer, a random subset of that size is taken. The subset can also be explicitly fixed by passing an Iterable with the corresponding indices.

        Returns:
            tensor: The local snapshots tensor of shape ``(T, n, 2, 2)`` containing the local local density matrices for each snapshot and each qubit.
        """
    if snapshots is not None:
        if isinstance(snapshots, int):
            pick_snapshots = np.random.choice(np.arange(snapshots, dtype=np.int64), size=snapshots, replace=False)
        else:
            pick_snapshots = snapshots
        pick_snapshots = qml.math.convert_like(pick_snapshots, self.bits)
        bits = qml.math.gather(self.bits, pick_snapshots)
        recipes = qml.math.gather(self.recipes, pick_snapshots)
    else:
        bits = self.bits
        recipes = self.recipes
    if isinstance(wires, Iterable):
        wires = qml.math.convert_like(wires, bits)
        bits = qml.math.T(qml.math.gather(qml.math.T(bits), wires))
        recipes = qml.math.T(qml.math.gather(qml.math.T(recipes), wires))
    T, n = bits.shape
    U = np.empty((T, n, 2, 2), dtype='complex')
    for i, u in enumerate(self.observables):
        U[np.where(recipes == i)] = u
    state = (qml.math.cast(1 - 2 * bits[:, :, None, None], np.complex64) * U + np.eye(2)) / 2
    return 3 * state - np.eye(2)[None, None, :, :]