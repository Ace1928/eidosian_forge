from typing import List, Optional
import httpx
import networkx as nx
from qcs_api_client.models import InstructionSetArchitecture
from qcs_api_client.operations.sync import get_instruction_set_architecture
from pyquil.api import QCSClientConfiguration
from pyquil.api._qcs_client import qcs_client
from pyquil.external.rpcq import CompilerISA
from pyquil.noise import NoiseModel
from pyquil.quantum_processor import AbstractQuantumProcessor
from pyquil.quantum_processor.transformers import qcs_isa_to_compiler_isa, qcs_isa_to_graph
def get_qcs_quantum_processor(quantum_processor_id: str, client_configuration: Optional[QCSClientConfiguration]=None, timeout: float=10.0) -> QCSQuantumProcessor:
    """
    Retrieve an instruction set architecture for the specified ``quantum_processor_id`` and initialize a
    ``QCSQuantumProcessor`` with it.

    :param quantum_processor_id: QCS ID for the quantum processor.
    :param timeout: Time limit for request, in seconds.
    :param client_configuration: Optional client configuration. If none is provided, a default one will
           be loaded.

    :return: A ``QCSQuantumProcessor`` with the requested ISA.
    """
    client_configuration = client_configuration or QCSClientConfiguration.load()
    with qcs_client(client_configuration=client_configuration, request_timeout=timeout) as client:
        isa = get_instruction_set_architecture(client=client, quantum_processor_id=quantum_processor_id).parsed
    return QCSQuantumProcessor(quantum_processor_id=quantum_processor_id, isa=isa)