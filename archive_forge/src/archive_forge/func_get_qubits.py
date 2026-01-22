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
def get_qubits(self, indices: bool=True) -> Set[QubitDesignator]:
    """
        Returns all of the qubit indices used in this program, including gate applications and
        allocated qubits. e.g.

            >>> p = Program()
            >>> p.inst(("H", 1))
            >>> p.get_qubits()
            {1}
            >>> q = QubitPlaceholder()
            >>> p.inst(H(q))
            >>> len(p.get_qubits())
            2

        :param indices: Return qubit indices as integers intead of the
            wrapping :py:class:`Qubit` object
        :return: A set of all the qubit indices used in this program
        """
    qubits: Set[QubitDesignator] = set()
    for instr in self.instructions:
        if isinstance(instr, (Gate, Measurement, ResetQubit, Pulse, Capture, RawCapture, ShiftFrequency, SetFrequency, SetPhase, ShiftPhase, SwapPhases, SetScale)):
            qubits |= instr.get_qubits(indices=indices)
    return qubits