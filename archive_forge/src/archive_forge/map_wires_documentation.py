from functools import partial
from typing import Callable, Union, Sequence
import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator
from pennylane.workflow import QNode
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript, QuantumTape
from pennylane import transform
Defines how matrix works if applied to a tape containing multiple operations.