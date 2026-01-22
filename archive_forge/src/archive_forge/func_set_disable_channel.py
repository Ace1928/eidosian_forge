from __future__ import annotations
from collections.abc import Iterator, Sequence
from copy import deepcopy
from enum import Enum
from functools import partial
from itertools import chain
import numpy as np
from qiskit import pulse
from qiskit.pulse.transforms import target_qobj_transform
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import events, types, drawings, device_info
from qiskit.visualization.pulse_v2.stylesheet import QiskitPulseStyle
def set_disable_channel(self, channel: pulse.channels.Channel, remove: bool=True):
    """Interface method to control visibility of pulse channels.

        Specified object in the blocked list will not be shown.

        Args:
            channel: A pulse channel object to disable.
            remove: Set `True` to disable, set `False` to enable.
        """
    if remove:
        self.disable_chans.add(channel)
    else:
        self.disable_chans.discard(channel)