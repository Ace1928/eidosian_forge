import copy
import datetime
from typing import Any, Iterable, Tuple, Union, Dict
import dateutil.parser
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.utils.units import apply_prefix
def is_qubit_operational(self, qubit: int) -> bool:
    """
        Return the operational status of the given qubit.

        Args:
            qubit: Qubit for which to return operational status of.

        Returns:
            Operational status of the given qubit.
        """
    properties = self.qubit_property(qubit)
    if 'operational' in properties:
        return bool(properties['operational'][0])
    return True