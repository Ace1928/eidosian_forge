import logging
import threading
from contextlib import contextmanager
from typing import Dict, Optional, cast, List, Iterator
import httpx
from qcs_api_client.client import QCSClientConfiguration
from qcs_api_client.models.translate_native_quil_to_encrypted_binary_request import (
from qcs_api_client.operations.sync import (
from qcs_api_client.types import UNSET
from rpcq.messages import ParameterSpec
from pyquil.api._abstract_compiler import AbstractCompiler, QuantumExecutable, EncryptedProgram
from pyquil.api._qcs_client import qcs_client
from pyquil.api._rewrite_arithmetic import rewrite_arithmetic
from pyquil.parser import parse_program, parse
from pyquil.quantum_processor import AbstractQuantumProcessor
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference, ExpressionDesignator
from pyquil.quilbase import Declare, Gate
def get_calibration_program(self, force_refresh: bool=False) -> Program:
    """
        Get the Quil-T calibration program associated with the underlying QPU.

        This will return a cached copy of the calibration program if present.
        Otherwise (or if forcing a refresh), it will fetch and store the
        calibration program from the QCS API.

        A calibration program contains a number of DEFCAL, DEFWAVEFORM, and
        DEFFRAME instructions. In sum, those instructions describe how a Quil-T
        program should be translated into analog instructions for execution on
        hardware.

        :param force_refresh: Whether or not to fetch a new calibration program before returning.
        :returns: A Program object containing the calibration definitions."""
    if force_refresh or self._calibration_program is None:
        try:
            self._calibration_program = self._fetch_calibration_program()
        except Exception as ex:
            raise RuntimeError('Could not fetch calibrations') from ex
    assert self._calibration_program is not None
    return self._calibration_program