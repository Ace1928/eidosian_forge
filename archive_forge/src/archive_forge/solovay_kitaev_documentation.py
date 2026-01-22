import math
import warnings
from functools import lru_cache
from scipy.spatial import KDTree
import pennylane as qml
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript
Recursive method as given in the Section 3 of arXiv:0505030