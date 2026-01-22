import warnings
from collections.abc import Iterable
from string import ascii_letters as ABC
import numpy as np
import pennylane as qml
def global_snapshots(self, wires=None, snapshots=None):
    """Compute the T x 2**n x 2**n global snapshots

        .. warning::

            Classical shadows are not intended to reconstruct global quantum states.
            This method requires exponential scaling of measurements for accurate representations. Further, the output scales exponentially in the output dimension,
            and is therefore not practical for larger systems. A warning is raised for systems of sizes ``n>16``.

        Args:
            wires (Iterable[int]): The wires over which to compute the snapshots. For ``wires=None`` (default) all ``n`` qubits are used.
            snapshots (Iterable[int] or int): Only compute a subset of local snapshots. For ``snapshots=None`` (default), all local snapshots are taken.
                In case of an integer, a random subset of that size is taken. The subset can also be explicitly fixed by passing an Iterable with the corresponding indices.

        Returns:
            tensor: The global snapshots tensor of shape ``(T, 2**n, 2**n)`` containing the density matrices for each snapshot measurement.

        **Example**

        We can approximately reconstruct a Bell state:

        .. code-block:: python3

            dev = qml.device("default.qubit", wires=range(2), shots=1000)
            @qml.qnode(dev)
            def qnode():
                qml.Hadamard(0)
                qml.CNOT((0,1))
                return classical_shadow(wires=range(2))

            bits, recipes = qnode()
            shadow = ClassicalShadow(bits, recipes)
            shadow_state = np.mean(shadow.global_snapshots(), axis=0)

            bell_state = np.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]])

        >>> np.allclose(bell_state, shadow_state, atol=1e-1)
        True

        """
    local_snapshot = self.local_snapshots(wires, snapshots)
    if local_snapshot.shape[1] > 16:
        warnings.warn('Querying density matrices for n_wires > 16 is not recommended, operation will take a long time', UserWarning)
    T, n = local_snapshot.shape[:2]
    transposed_snapshots = np.transpose(local_snapshot, axes=(1, 0, 2, 3))
    old_indices = [f'a{ABC[1 + 2 * i:3 + 2 * i]}' for i in range(n)]
    new_indices = f'a{ABC[1:2 * n + 1:2]}{ABC[2:2 * n + 1:2]}'
    return np.reshape(np.einsum(f'{','.join(old_indices)}->{new_indices}', *transposed_snapshots), (T, 2 ** n, 2 ** n))