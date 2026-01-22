import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.MottonenStatePreparation.decomposition`.

        Args:
            state_vector (tensor_like): Normalized state vector of shape ``(2^len(wires),)``
            wires (Any or Iterable[Any]): wires that the operator acts on

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        >>> state_vector = torch.tensor([0.5, 0.5, 0.5, 0.5])
        >>> qml.MottonenStatePreparation.compute_decomposition(state_vector, wires=["a", "b"])
        [RY(array(1.57079633), wires=['a']),
        RY(array(1.57079633), wires=['b']),
        CNOT(wires=['a', 'b']),
        CNOT(wires=['a', 'b'])]
        