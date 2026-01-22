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
def get_rigetti_qcs_service(quantum_processor_id: str, *, as_qvm: Optional[bool]=None, noisy: Optional[bool]=None, executor: executors.CircuitSweepExecutor=_default_executor, transformer: transformers.CircuitTransformer=transformers.default) -> RigettiQCSService:
    """Calls `pyquil.get_qc` to initialize a `pyquil.api.QuantumComputer` and uses
    this to initialize `RigettiQCSService`.

    Args:
        quantum_processor_id: The name of the desired quantum computer. This should
            correspond to a name returned by `pyquil.api.list_quantum_computers`. Names
            ending in "-qvm" will return a QVM. Names ending in "-pyqvm" will return a
            `pyquil.PyQVM`. Otherwise, we will return a Rigetti QCS QPU if one exists
            with the requested name.
        as_qvm: An optional flag to force construction of a QVM (instead of a QPU). If
            specified and set to `True`, a QVM-backed quantum computer will be returned regardless
            of the name's suffix
        noisy: An optional flag to force inclusion of a noise model. If
            specified and set to `True`, a quantum computer with a noise model will be returned.
            The generic QVM noise model is simple T1 and T2 noise plus readout error. At the time
            of this writing, this has no effect on a QVM initialized based on a Rigetti QCS
            `qcs_api_client.models.InstructionSetArchitecture`.
        executor: A callable that first uses the below transformer on cirq.Circuit s and
            then executes the transformed circuit on the quantum_computer. You may pass your
            own callable or any static method on CircuitSweepExecutors.
        transformer: A callable that transforms the cirq.Circuit into a pyquil.Program.
            You may pass your own callable or any static method on CircuitTransformers.

    Returns:
        A `RigettiQCSService` with the specified quantum processor, executor, and transformer.

    """
    qc = get_qc(quantum_processor_id, as_qvm=as_qvm, noisy=noisy)
    return RigettiQCSService(quantum_computer=qc, executor=executor, transformer=transformer)