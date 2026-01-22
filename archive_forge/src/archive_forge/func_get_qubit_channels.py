import re
import copy
import numbers
from typing import Dict, List, Any, Iterable, Tuple, Union
from collections import defaultdict
from qiskit.exceptions import QiskitError
from qiskit.providers.exceptions import BackendConfigurationError
from qiskit.pulse.channels import (
def get_qubit_channels(self, qubit: Union[int, Iterable[int]]) -> List[Channel]:
    """Return a list of channels which operate on the given ``qubit``.

        Raises:
            BackendConfigurationError: If ``qubit`` is not a found or if
                the backend does not provide `channels` information in its configuration.

        Returns:
            List of ``Channel``\\s operated on my the given ``qubit``.
        """
    channels = set()
    try:
        if isinstance(qubit, int):
            for key in self._qubit_channel_map.keys():
                if qubit in key:
                    channels.update(self._qubit_channel_map[key])
            if len(channels) == 0:
                raise KeyError
        elif isinstance(qubit, list):
            qubit = tuple(qubit)
            channels.update(self._qubit_channel_map[qubit])
        elif isinstance(qubit, tuple):
            channels.update(self._qubit_channel_map[qubit])
        return list(channels)
    except KeyError as ex:
        raise BackendConfigurationError(f"Couldn't find the qubit - {qubit}") from ex
    except AttributeError as ex:
        raise BackendConfigurationError(f"This backend - '{self.backend_name}' does not provide channel information.") from ex