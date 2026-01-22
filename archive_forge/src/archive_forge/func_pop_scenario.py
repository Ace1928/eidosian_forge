import copy
import itertools
from collections import OrderedDict
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.transforms import transform
from pennylane import adjoint
from pennylane.ops.qubit.attributes import symmetric_over_all_wires
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.transforms.commutation_dag import commutation_dag
from pennylane.wires import Wires
def pop_scenario(self):
    """
        Pop the first scenario of the list.
        Returns:
            MatchingScenarios: a scenario of match.
        """
    first = self.matching_scenarios_list[0]
    self.matching_scenarios_list.pop(0)
    return first