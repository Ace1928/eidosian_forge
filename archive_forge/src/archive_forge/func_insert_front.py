from functools import partial
from typing import Callable, List, Tuple, Optional, Sequence, Union
import pennylane as qml
from pennylane.typing import Result, ResultBatch
from pennylane.tape import QuantumTape
from .transform_dispatcher import TransformContainer, TransformError, TransformDispatcher
def insert_front(self, transform_container: TransformContainer):
    """Insert the transform container at the beginning of the program.

        Args:
            transform_container(TransformContainer): A transform represented by its container.
        """
    if transform_container.final_transform and (not self.is_empty()):
        raise TransformError('Informative transforms can only be added at the end of the program.')
    self._transform_program.insert(0, transform_container)