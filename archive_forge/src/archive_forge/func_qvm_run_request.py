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
def qvm_run_request(quil_program: Program, classical_addresses: Mapping[str, Union[bool, Sequence[int]]], trials: int, measurement_noise: Optional[Tuple[float, float, float]], gate_noise: Optional[Tuple[float, float, float]], random_seed: Optional[int]) -> RunProgramRequest:
    if not quil_program:
        raise ValueError('You have attempted to run an empty program. Please provide gates or measure instructions to your program.')
    if not isinstance(quil_program, Program):
        raise TypeError('quil_program must be a Quil program object')
    classical_addresses = prepare_register_list(classical_addresses)
    if not isinstance(trials, int):
        raise TypeError('trials must be an integer')
    return RunProgramRequest(program=quil_program.out(calibrations=False), addresses=classical_addresses, trials=trials, measurement_noise=measurement_noise, gate_noise=gate_noise, seed=random_seed)