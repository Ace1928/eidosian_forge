from math import exp
from typing import Dict, Optional, Sequence, TYPE_CHECKING
import cirq
from cirq.devices.noise_model import validate_all_measurements
A depolarizing noise model with damped readout error.

        All error modes are specified on a per-qubit basis. To omit a given
        error mode from the noise model, leave its dict blank when initializing
        this object.

        Args:
            depol_probs: Dict of depolarizing probabilities for each qubit.
            bitflip_probs: Dict of bit-flip probabilities during measurement.
            decay_probs: Dict of T1 decay probabilities during measurement.
                Bitflip noise is applied first, then amplitude decay.
        