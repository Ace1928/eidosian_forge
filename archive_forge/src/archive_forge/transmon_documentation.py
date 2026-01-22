import warnings
from dataclasses import dataclass
from typing import Callable, List, Union
import numpy as np
import pennylane as qml
from pennylane.pulse import HardwareHamiltonian
from pennylane.pulse.hardware_hamiltonian import HardwarePulse
from pennylane.typing import TensorLike
from pennylane.wires import Wires
Takes `params`, and reorganizes it based on whether the Hamiltonian has
    callable phase and/or callable amplitude and/or callable freq.

    Consolidates amplitude, phase and freq parameters if they are callable,
    and duplicates parameters since they will be passed to two operators in the Hamiltonian