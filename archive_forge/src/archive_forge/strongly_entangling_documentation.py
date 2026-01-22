import pennylane as qml
from pennylane.operation import Operation, AnyWires
Returns the expected shape of the weights tensor.

        Args:
            n_layers (int): number of layers
            n_wires (int): number of wires

        Returns:
            tuple[int]: shape
        