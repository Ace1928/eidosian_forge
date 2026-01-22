import abc
from typing import Callable, List
import copy
import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, _UNSET_BATCH_SIZE
from pennylane.wires import Wires
@property
def overlapping_ops(self) -> List[List[Operator]]:
    """Groups all operands of the composite operator that act on overlapping wires.

        Returns:
            List[List[Operator]]: List of lists of operators that act on overlapping wires. All the
            inner lists commute with each other.
        """
    if self._overlapping_ops is None:
        overlapping_ops = []
        for op in self:
            ops = [op]
            wires = op.wires
            op_added = False
            for idx, (old_wires, old_ops) in enumerate(overlapping_ops):
                if any((wire in old_wires for wire in wires)):
                    overlapping_ops[idx] = (old_wires + wires, old_ops + ops)
                    op_added = True
                    break
            if not op_added:
                overlapping_ops.append((op.wires, [op]))
        self._overlapping_ops = [overlapping_op[1] for overlapping_op in overlapping_ops]
    return self._overlapping_ops