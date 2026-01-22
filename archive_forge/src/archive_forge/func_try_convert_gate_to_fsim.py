import abc
import collections
import dataclasses
import functools
import math
import re
from typing import (
import numpy as np
import pandas as pd
import cirq
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.engine import (
from cirq_google.ops import FSimGateFamily, SycamoreGate
def try_convert_gate_to_fsim(gate: cirq.Gate) -> Optional[PhaseCalibratedFSimGate]:
    """Converts a gate to equivalent PhaseCalibratedFSimGate if possible.

    Args:
        gate: Gate to convert.

    Returns:
        If provided gate is equivalent to some PhaseCalibratedFSimGate, returns that gate.
        Otherwise returns None.
    """
    cgate = FSimGateFamily().convert(gate, cirq.PhasedFSimGate)
    if cgate is None or cirq.is_parameterized(cgate):
        return None
    assert isinstance(cgate.zeta, float) and isinstance(cgate.gamma, float)
    if not (np.isclose(cgate.zeta, 0.0) and np.isclose(cgate.gamma, 0.0)):
        return None
    theta = cgate.theta
    phi = cgate.phi
    phase_exponent = -cgate.chi / (2 * np.pi)
    phi = phi % (2 * np.pi)
    theta = theta % (2 * np.pi)
    if theta >= np.pi:
        theta = 2 * np.pi - theta
        phase_exponent = phase_exponent + 0.5
    phase_exponent %= 1
    return PhaseCalibratedFSimGate(cirq.FSimGate(theta=theta, phi=phi), phase_exponent)