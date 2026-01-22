from collections import defaultdict
from typing import List, Dict, Any, Tuple, Iterator, Optional, Union
import numpy as np
from qiskit import pulse
from qiskit.visualization.pulse_v2 import types
from qiskit.visualization.pulse_v2.device_info import DrawerBackendInfo
def qubit_index_sort(channels: List[pulse.channels.Channel], formatter: Dict[str, Any], device: DrawerBackendInfo) -> Iterator[Tuple[str, List[pulse.channels.Channel]]]:
    """Layout function for the channel assignment to the chart instance.

    Assign multiple channels per chart. Channels associated with the same qubit
    are grouped in the same chart and sorted by qubit index in ascending order.

    Acquire channels are not shown.

    Stylesheet key:
        `chart_channel_map`

    For example:
        [D0, D2, C0, C2, M0, M2, A0, A2] -> [Q0, Q1, Q2]

    Args:
        channels: Channels to show.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Yields:
        Tuple of chart name and associated channels.
    """
    _removed = (pulse.channels.AcquireChannel, pulse.channels.MemorySlot, pulse.channels.RegisterSlot)
    qubit_channel_map = defaultdict(list)
    for chan in channels:
        if isinstance(chan, _removed):
            continue
        qubit_channel_map[device.get_qubit_index(chan)].append(chan)
    sorted_map = sorted(qubit_channel_map.items(), key=lambda x: x[0])
    for qind, chans in sorted_map:
        yield (f'Q{qind:d}', chans)