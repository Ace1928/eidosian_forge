from typing import Union
from functools import lru_cache
import numpy as np
from qiskit.circuit import Instruction
from qiskit.pulse import Schedule, ScheduleBlock, builder, ScalableSymbolicPulse
from qiskit.pulse.channels import Channel
from qiskit.pulse.library.symbolic_pulses import Drag
from qiskit.transpiler.passes.calibration.base_builder import CalibrationBuilder
from qiskit.transpiler import Target
from qiskit.circuit.library.standard_gates import RXGate
from qiskit.exceptions import QiskitError

        Generate RX calibration for the rotation angle specified in node_op.
        