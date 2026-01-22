from typing import Sequence, Callable
from functools import reduce
import pennylane as qml
from pennylane.transforms import transform
re-order the output to the original shape and order