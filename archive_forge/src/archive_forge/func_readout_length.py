import copy
import datetime
from typing import Any, Iterable, Tuple, Union, Dict
import dateutil.parser
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.utils.units import apply_prefix
def readout_length(self, qubit: int) -> float:
    """
        Return the readout length [sec] of the given qubit.

        Args:
            qubit: Qubit for which to return the readout length of.

        Return:
            Readout length of the given qubit.
        """
    return self.qubit_property(qubit, 'readout_length')[0]