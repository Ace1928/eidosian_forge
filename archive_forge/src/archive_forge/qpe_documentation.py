import pennylane as qml
from pennylane.queuing import QueuingManager
from pennylane.operation import AnyWires, Operation, Operator
Representation of the QPE circuit as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.QuantumPhaseEstimation.decomposition`.

        Args:
            wires (Any or Iterable[Any]): wires that the QPE circuit acts on
            unitary (Operator): the phase estimation unitary, specified as an operator
            target_wires (Any or Iterable[Any]): the target wires to apply the unitary
            estimation_wires (Any or Iterable[Any]): the wires to be used for phase estimation

        Returns:
            list[.Operator]: decomposition of the operator
        