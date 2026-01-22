import itertools
import types
import warnings
from collections import defaultdict
from typing import (
import numpy as np
from rpcq.messages import NativeQuilMetadata, ParameterAref
from pyquil._parser.parser import run_parser
from pyquil._memory import Memory
from pyquil.gates import MEASURE, RESET, MOVE
from pyquil.noise import _check_kraus_ops, _create_kraus_pragmas, pauli_kraus_map
from pyquil.quilatom import (
from pyquil.quilbase import (
from pyquil.quiltcalibrations import (
def copy_everything_except_instructions(self) -> 'Program':
    """
        Copy all the members that live on a Program object.

        :return: a new Program
        """
    new_prog = Program()
    new_prog._calibrations = self.calibrations.copy()
    new_prog._declarations = self._declarations.copy()
    new_prog._waveforms = self.waveforms.copy()
    new_prog._defined_gates = self._defined_gates.copy()
    new_prog._frames = self.frames.copy()
    if self.native_quil_metadata is not None:
        new_prog.native_quil_metadata = self.native_quil_metadata.copy()
    new_prog.num_shots = self.num_shots
    new_prog._memory = self._memory.copy()
    return new_prog