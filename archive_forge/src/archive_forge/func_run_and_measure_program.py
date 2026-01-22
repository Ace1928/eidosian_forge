import re
from contextlib import contextmanager
from dataclasses import dataclass
from json.decoder import JSONDecodeError
from typing import Iterator, Any, Dict, Union, Tuple, Optional, List, cast
import httpx
from qcs_api_client.client import QCSClientConfiguration
from pyquil.api._errors import ApiError, UnknownApiError, TooManyQubitsError, error_mapping
def run_and_measure_program(self, request: RunAndMeasureProgramRequest) -> RunAndMeasureProgramResponse:
    """
        Run and measure a Quil program, and return its results.
        """
    payload: Dict[str, Any] = {'type': 'multishot-measure', 'compiled-quil': request.program, 'qubits': request.qubits, 'trials': request.trials}
    if request.measurement_noise is not None:
        payload['measurement-noise'] = request.measurement_noise
    if request.gate_noise is not None:
        payload['gate-noise'] = request.gate_noise
    if request.seed is not None:
        payload['rng-seed'] = request.seed
    return RunAndMeasureProgramResponse(results=cast(List[List[int]], self._post_json(payload).json()))