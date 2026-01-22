from itertools import combinations
import pennylane as qml
from pennylane.operation import Operation, AnyWires
Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.IQPEmbedding.decomposition`.

        Args:
            features (tensor_like): tensor of features to encode
            wires (Any or Iterable[Any]): wires that the template acts on

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        >>> features = torch.tensor([1., 2., 3.])
        >>> pattern = [(0, 1), (0, 2), (1, 2)]
        >>> qml.IQPEmbedding.compute_decomposition(features, wires=[0, 1, 2], n_repeats=2, pattern=pattern)
        [Hadamard(wires=[0]), RZ(tensor(1.), wires=[0]),
         Hadamard(wires=[1]), RZ(tensor(2.), wires=[1]),
         Hadamard(wires=[2]), RZ(tensor(3.), wires=[2]),
         MultiRZ(tensor(2.), wires=[0, 1]), MultiRZ(tensor(3.), wires=[0, 2]), MultiRZ(tensor(6.), wires=[1, 2]),
         Hadamard(wires=[0]), RZ(tensor(1.), wires=[0]),
         Hadamard(wires=[1]), RZ(tensor(2.), wires=[1]),
         Hadamard(wires=[2]), RZ(tensor(3.), wires=[2]),
         MultiRZ(tensor(2.), wires=[0, 1]), MultiRZ(tensor(3.), wires=[0, 2]), MultiRZ(tensor(6.), wires=[1, 2])]
        