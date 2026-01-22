from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Union, Optional
from qiskit import pulse
from qiskit.providers import BackendConfigurationError
from qiskit.providers.backend import Backend
def get_qubit_index(self, chan: pulse.channels.Channel) -> Union[int, None]:
    """Get associated qubit index of given channel object."""
    for qind, chans in self._qubit_channel_map.items():
        if chan in chans:
            return qind
    return chan.index