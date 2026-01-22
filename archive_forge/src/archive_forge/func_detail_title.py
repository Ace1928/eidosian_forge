from collections import defaultdict
from typing import List, Dict, Any, Tuple, Iterator, Optional, Union
import numpy as np
from qiskit import pulse
from qiskit.visualization.pulse_v2 import types
from qiskit.visualization.pulse_v2.device_info import DrawerBackendInfo
def detail_title(program: Union[pulse.Waveform, pulse.Schedule], device: DrawerBackendInfo) -> str:
    """Layout function for generating figure title.

    This layout writes program name, program duration, and backend name in the title.
    """
    title_str = []
    title_str.append(f'Name: {program.name}')
    dt = device.dt * 1000000000.0 if device.dt else 1.0
    title_str.append('Duration: {dur:.1f} {unit}'.format(dur=program.duration * dt, unit='ns' if device.dt else 'dt'))
    if device.backend_name != 'no-backend':
        title_str.append(f'Backend: {device.backend_name}')
    return ', '.join(title_str)