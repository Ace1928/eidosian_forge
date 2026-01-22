from abc import ABC, abstractmethod
from dataclasses import dataclass
import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Union
from pyquil._memory import Memory
from pyquil._version import pyquil_version
from pyquil.api._compiler_client import CompilerClient, CompileToNativeQuilRequest
from pyquil.external.rpcq import compiler_isa_to_target_quantum_processor
from pyquil.parser import parse_program
from pyquil.paulis import PauliTerm
from pyquil.quantum_processor import AbstractQuantumProcessor
from pyquil.quil import Program
from pyquil.quilatom import ExpressionDesignator, MemoryReference
from pyquil.quilbase import Gate
from qcs_api_client.client import QCSClientConfiguration
from rpcq.messages import NativeQuilMetadata, ParameterAref, ParameterSpec
def quil_to_native_quil(self, program: Program, *, protoquil: Optional[bool]=None) -> Program:
    """
        Compile an arbitrary quil program according to the ISA of ``self.quantum_processor``.

        :param program: Arbitrary quil to compile
        :param protoquil: Whether to restrict to protoquil (``None`` means defer to server)
        :return: Native quil and compiler metadata
        """
    self._connect()
    compiler_isa = self.quantum_processor.to_compiler_isa()
    request = CompileToNativeQuilRequest(program=program.out(calibrations=False), target_quantum_processor=compiler_isa_to_target_quantum_processor(compiler_isa), protoquil=protoquil)
    response = self._compiler_client.compile_to_native_quil(request)
    nq_program = parse_program(response.native_program)
    nq_program.native_quil_metadata = None if response.metadata is None else NativeQuilMetadata(final_rewiring=response.metadata.final_rewiring, gate_depth=response.metadata.gate_depth, gate_volume=response.metadata.gate_volume, multiqubit_gate_depth=response.metadata.multiqubit_gate_depth, program_duration=response.metadata.program_duration, program_fidelity=response.metadata.program_fidelity, topological_swaps=response.metadata.topological_swaps, qpu_runtime_estimation=response.metadata.qpu_runtime_estimation)
    nq_program.num_shots = program.num_shots
    nq_program._calibrations = program.calibrations
    nq_program._memory = program._memory.copy()
    return nq_program