from __future__ import annotations
import enum
import warnings
from collections.abc import Sequence
from math import pi, erf
import numpy as np
from qiskit.circuit import Instruction as CircuitInst
from qiskit.circuit.library.standard_gates import RZXGate
from qiskit.exceptions import QiskitError
from qiskit.pulse import (
from qiskit.pulse import builder
from qiskit.pulse.filters import filter_instructions
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.transpiler.target import Target
from .base_builder import CalibrationBuilder
from .exceptions import CalibrationNotAvailable
@staticmethod
@builder.macro
def rescale_cr_inst(instruction: Play, theta: float, sample_mult: int=16) -> int:
    """A builder macro to play stretched pulse.

        Args:
            instruction: The instruction from which to create a new shortened or lengthened pulse.
            theta: desired angle, pi/2 is assumed to be the angle that the pulse in the given
                play instruction implements.
            sample_mult: All pulses must be a multiple of sample_mult.

        Returns:
            Duration of stretched pulse.

        Raises:
            QiskitError: if rotation angle is not assigned.
        """
    try:
        theta = float(theta)
    except TypeError as ex:
        raise QiskitError('Target rotation angle is not assigned.') from ex
    params = instruction.pulse.parameters.copy()
    risefall_sigma_ratio = (params['duration'] - params['width']) / params['sigma']
    risefall_area = params['sigma'] * np.sqrt(2 * pi) * erf(risefall_sigma_ratio)
    full_area = params['width'] + risefall_area
    cal_angle = pi / 2
    target_area = abs(theta) / cal_angle * full_area
    new_width = target_area - risefall_area
    if new_width >= 0:
        width = new_width
        params['amp'] *= np.sign(theta)
    else:
        width = 0
        params['amp'] *= np.sign(theta) * target_area / risefall_area
    round_duration = round((width + risefall_sigma_ratio * params['sigma']) / sample_mult) * sample_mult
    params['duration'] = round_duration
    params['width'] = width
    stretched_pulse = GaussianSquare(**params)
    builder.play(stretched_pulse, instruction.channel)
    return round_duration