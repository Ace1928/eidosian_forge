import contextlib
import copy
from collections import Counter
from typing import List, Union, Optional, Sequence
import pennylane as qml
from pennylane.measurements import (
from pennylane.typing import TensorLike
from pennylane.operation import Observable, Operator, Operation, _UNSET_BATCH_SIZE
from pennylane.pytrees import register_pytree
from pennylane.queuing import AnnotatedQueue, process_queue
from pennylane.wires import Wires
@property
def shots(self) -> Shots:
    """Returns a ``Shots`` object containing information about the number
        and batches of shots

        Returns:
            ~.Shots: Object with shot information
        """
    return self._shots