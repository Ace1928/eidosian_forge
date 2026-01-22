from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import cirq
from cirq import ops, linalg, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker
Operates on target qubits in a way that varies based on an input qureg.