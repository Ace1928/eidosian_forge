from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Union, Tuple
import numpy as np
from qcs_api_client.client import QCSClientConfiguration
from pyquil._version import pyquil_version
from pyquil.api import QuantumExecutable
from pyquil.api._qam import QAM, QAMExecutionResult
from pyquil.api._qvm_client import (
from pyquil.noise import NoiseModel, apply_noise_model
from pyquil.quil import Program, get_classical_addresses_from_program
def validate_noise_probabilities(noise_parameter: Optional[Tuple[float, float, float]]) -> None:
    """
    Is noise_parameter a valid specification of noise probabilities for depolarizing noise?

    :param tuple noise_parameter: Tuple of 3 noise parameter values to be validated.
    """
    if not noise_parameter:
        return
    if not isinstance(noise_parameter, tuple):
        raise TypeError('noise_parameter must be a tuple')
    if any([not isinstance(value, float) for value in noise_parameter]):
        raise TypeError('noise_parameter values should all be floats')
    if len(noise_parameter) != 3:
        raise ValueError('noise_parameter tuple must be of length 3')
    if sum(noise_parameter) > 1 or sum(noise_parameter) < 0:
        raise ValueError('sum of entries in noise_parameter must be between 0 and 1 (inclusive)')
    if any([value < 0 for value in noise_parameter]):
        raise ValueError('noise_parameter values should all be non-negative')