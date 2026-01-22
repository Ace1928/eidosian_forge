import abc
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import protocols, value
from cirq.type_workarounds import NotImplementedType
@property
def classical_data(self) -> 'cirq.ClassicalDataStoreReader':
    return self._classical_data