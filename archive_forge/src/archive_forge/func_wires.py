from abc import abstractmethod
from copy import copy
import numpy as np
import pennylane as qml
from pennylane.operation import Operator, _UNSET_BATCH_SIZE
from pennylane.queuing import QueuingManager
@property
def wires(self):
    return self.base.wires