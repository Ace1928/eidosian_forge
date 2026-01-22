from copy import copy
import numpy as np
import pennylane as qml
from pennylane.ops import Prod, SProd
from pennylane.pauli.utils import (
from pennylane.wires import Wires
from .graph_colouring import largest_first, recursive_largest_first

        Runs the graph colouring heuristic algorithm to obtain the partitioned Pauli words.

        Returns:
            list[list[Observable]]: a list of the obtained groupings. Each grouping is itself a
            list of Pauli word ``Observable`` instances
        