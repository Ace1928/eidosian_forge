import re
import copy
import numbers
from typing import Dict, List, Any, Iterable, Tuple, Union
from collections import defaultdict
from qiskit.exceptions import QiskitError
from qiskit.providers.exceptions import BackendConfigurationError
from qiskit.pulse.channels import (
def get_channel_qubits(self, channel: Channel) -> List[int]:
    """
        Return a list of indices for qubits which are operated on directly by the given ``channel``.

        Raises:
            BackendConfigurationError: If ``channel`` is not a found or if
                the backend does not provide `channels` information in its configuration.

        Returns:
            List of qubits operated on my the given ``channel``.
        """
    try:
        return self._channel_qubit_map[channel]
    except KeyError as ex:
        raise BackendConfigurationError(f"Couldn't find the Channel - {channel}") from ex
    except AttributeError as ex:
        raise BackendConfigurationError(f"This backend - '{self.backend_name}' does not provide channel information.") from ex