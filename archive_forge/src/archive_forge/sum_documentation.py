import warnings
import itertools
from copy import copy
from typing import List
import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, convert_to_opmath
from pennylane.ops.qubit import Hamiltonian
from pennylane.queuing import QueuingManager
from .composite import CompositeOp
Sorting key used in the `sorted` python built-in function.

            Args:
                op (.Operator): Operator.

            Returns:
                Tuple[int, int, str]: Tuple containing the minimum wire value, the number of wires
                    and the string of the operator. This tuple is used to compare different operators
                    in the sorting algorithm.
            