from functools import partial
from typing import Callable, List, Tuple, Optional, Sequence, Union
import pennylane as qml
from pennylane.typing import Result, ResultBatch
from pennylane.tape import QuantumTape
from .transform_dispatcher import TransformContainer, TransformError, TransformDispatcher
def insert_front_transform(self, transform: TransformDispatcher, *targs, **tkwargs):
    """Add a transform (dispatcher) to the beginning of the program.

        Args:
            transform(TransformDispatcher): The transform to add to the front of the transform program.
            *targs: Any additional arguments that are passed to the transform.

        Keyword Args:
            **tkwargs: Any additional keyword arguments that are passed to the transform.

        """
    if transform.final_transform and (not self.is_empty()):
        raise TransformError('Informative transforms can only be added at the end of the program.')
    self.insert_front(TransformContainer(transform.transform, targs, tkwargs, transform.classical_cotransform, transform.is_informative, transform.final_transform))
    if transform.expand_transform:
        self.insert_front(TransformContainer(transform.expand_transform, targs, tkwargs))