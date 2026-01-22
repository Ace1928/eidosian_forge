import copy
import datetime
from typing import Any, Iterable, Tuple, Union, Dict
import dateutil.parser
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.utils.units import apply_prefix
def qubit_property(self, qubit: int, name: str=None) -> Union[Dict[str, PropertyT], PropertyT]:
    """
        Return the property of the given qubit.

        Args:
            qubit: The property to look for.
            name: Optionally used to specify within the hierarchy which property to return.

        Returns:
            Qubit property as a tuple of the value and the time it was measured.

        Raises:
            BackendPropertyError: If the property is not found.
        """
    try:
        result = self._qubits[qubit]
        if name is not None:
            result = result[name]
    except KeyError as ex:
        raise BackendPropertyError("Couldn't find the propert{name} for qubit {qubit}.".format(name="y '" + name + "'" if name else 'ies', qubit=qubit)) from ex
    return result