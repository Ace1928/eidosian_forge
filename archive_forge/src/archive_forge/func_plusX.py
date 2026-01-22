import logging
import re
from dataclasses import dataclass
from typing import Any, FrozenSet, Generator, Iterable, List, Optional, cast
from pyquil.paulis import PauliTerm, sI
def plusX(q: int) -> TensorProductState:
    return TensorProductState([_OneQState(label='X', index=0, qubit=q)])