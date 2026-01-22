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
def match_calibrations(self, instr: AbstractInstruction) -> Optional[CalibrationMatch]:
    """
        Attempt to match a calibration to the provided instruction.

        Note: preference is given to later calibrations, i.e. in a program with

          DEFCAL X 0:
              <a>

          DEFCAL X 0:
             <b>

        the second calibration, with body <b>, would be the calibration matching `X 0`.

        :param instr: An instruction.
        :returns: a CalibrationMatch object, if one can be found.
        """
    if isinstance(instr, (Gate, Measurement)):
        for cal in reversed(self.calibrations):
            match = match_calibration(instr, cal)
            if match is not None:
                return match
    return None