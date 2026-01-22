from typing import cast, Optional
import cirq
import httpx
from pyquil import get_qc
from qcs_api_client.operations.sync import (
from qcs_api_client.models import (
from pyquil.api import QuantumComputer
from cirq_rigetti.sampler import RigettiQCSSampler
from cirq_rigetti._qcs_api_client_decorator import _provide_default_client
from cirq_rigetti import circuit_transformers as transformers
from cirq_rigetti import circuit_sweep_executors as executors
@staticmethod
@_provide_default_client
def get_quilt_calibrations(quantum_processor_id: str, client: Optional[httpx.Client]) -> GetQuiltCalibrationsResponse:
    """Retrieve the calibration data used for client-side Quil-T generation.

        Args:
            quantum_processor_id: The identifier of the Rigetti QCS quantum processor.
            client: Optional; A `httpx.Client` initialized with Rigetti QCS credentials
            and configuration. If not provided, `qcs_api_client` will initialize a
            configured client based on configured values in the current user's
            `~/.qcs` directory or default values.

        Returns:
            A qcs_api_client.models.GetQuiltCalibrationsResponse containing the
            device calibrations.
        """
    return cast(GetQuiltCalibrationsResponse, get_quilt_calibrations(client=client, quantum_processor_id=quantum_processor_id).parsed)