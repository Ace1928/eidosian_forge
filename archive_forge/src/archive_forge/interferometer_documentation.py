from itertools import product
import pennylane as qml
from pennylane.ops import Beamsplitter, Rotation
from pennylane.wires import Wires
from pennylane.operation import CVOperation, AnyWires
Returns a list of shapes for the 3 parameter tensors.

        Args:
            n_wires (int): number of wires

        Returns:
            list[tuple[int]]: list of shapes
        