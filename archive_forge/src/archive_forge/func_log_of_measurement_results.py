import abc
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import protocols, value
from cirq.type_workarounds import NotImplementedType
@property
def log_of_measurement_results(self) -> Dict[str, List[int]]:
    """Gets the log of measurement results."""
    return {str(k): list(self.classical_data.get_digits(k)) for k in self.classical_data.keys()}