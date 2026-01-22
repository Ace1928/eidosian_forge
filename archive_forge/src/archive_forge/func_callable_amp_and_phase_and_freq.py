import warnings
from dataclasses import dataclass
from typing import Callable, List, Union
import numpy as np
import pennylane as qml
from pennylane.pulse import HardwareHamiltonian
from pennylane.pulse.hardware_hamiltonian import HardwarePulse
from pennylane.typing import TensorLike
from pennylane.wires import Wires
def callable_amp_and_phase_and_freq(params, t):
    return hz_to_rads * amp(params[0], t) * trig_fn(phase(params[1], t) + hz_to_rads * freq(params[2], t) * t)