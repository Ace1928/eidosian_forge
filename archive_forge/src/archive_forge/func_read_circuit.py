from collections import defaultdict
import io
import json
import struct
import uuid
import warnings
import numpy as np
from qiskit import circuit as circuit_mod
from qiskit.circuit import library, controlflow, CircuitInstruction, ControlFlowOp
from qiskit.circuit.classical import expr
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.gate import Gate
from qiskit.circuit.singleton import SingletonInstruction, SingletonGate
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.annotated_operation import (
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.qpy import common, formats, type_keys
from qiskit.qpy.binary_io import value, schedules
from qiskit.quantum_info.operators import SparsePauliOp, Clifford
from qiskit.synthesis import evolution as evo_synth
from qiskit.transpiler.layout import Layout, TranspileLayout
def read_circuit(file_obj, version, metadata_deserializer=None, use_symengine=False):
    """Read a single QuantumCircuit object from the file like object.

    Args:
        file_obj (FILE): The file like object to read the circuit data from.
        version (int): QPY version.
        metadata_deserializer (JSONDecoder): An optional JSONDecoder class
            that will be used for the ``cls`` kwarg on the internal
            ``json.load`` call used to deserialize the JSON payload used for
            the :attr:`.QuantumCircuit.metadata` attribute for a circuit
            in the file-like object. If this is not specified the circuit metadata will
            be parsed as JSON with the stdlib ``json.load()`` function using
            the default ``JSONDecoder`` class.
        use_symengine (bool): If True, symbolic objects will be de-serialized using
            symengine's native mechanism. This is a faster serialization alternative, but not
            supported in all platforms. Please check that your target platform is supported by
            the symengine library before setting this option, as it will be required by qpy to
            deserialize the payload.
    Returns:
        QuantumCircuit: The circuit object from the file.

    Raises:
        QpyError: Invalid register.
    """
    vectors = {}
    if version < 2:
        header, name, metadata = _read_header(file_obj, metadata_deserializer=metadata_deserializer)
    else:
        header, name, metadata = _read_header_v2(file_obj, version, vectors, metadata_deserializer=metadata_deserializer)
    global_phase = header['global_phase']
    num_qubits = header['num_qubits']
    num_clbits = header['num_clbits']
    num_registers = header['num_registers']
    num_instructions = header['num_instructions']
    out_registers = {'q': {}, 'c': {}}
    all_registers = []
    out_bits = {'q': [None] * num_qubits, 'c': [None] * num_clbits}
    if num_registers > 0:
        if version < 4:
            registers = _read_registers(file_obj, num_registers)
        else:
            registers = _read_registers_v4(file_obj, num_registers)
        for bit_type_label, bit_type, reg_type in [('q', Qubit, QuantumRegister), ('c', Clbit, ClassicalRegister)]:
            typed_bits = out_bits[bit_type_label]
            typed_registers = registers[bit_type_label]
            for register_name, (standalone, indices, _incircuit) in typed_registers.items():
                if not standalone:
                    continue
                register = reg_type(len(indices), register_name)
                out_registers[bit_type_label][register_name] = register
                for owned, index in zip(register, indices):
                    if index >= 0:
                        typed_bits[index] = owned
            typed_bits = [bit if bit is not None else bit_type() for bit in typed_bits]
            for register_name, (standalone, indices, in_circuit) in typed_registers.items():
                if standalone:
                    register = out_registers[bit_type_label][register_name]
                else:
                    register = reg_type(name=register_name, bits=[typed_bits[x] if x >= 0 else bit_type() for x in indices])
                    out_registers[bit_type_label][register_name] = register
                if in_circuit:
                    all_registers.append(register)
            out_bits[bit_type_label] = typed_bits
    else:
        out_bits = {'q': [Qubit() for _ in out_bits['q']], 'c': [Clbit() for _ in out_bits['c']]}
    circ = QuantumCircuit(out_bits['q'], out_bits['c'], *all_registers, name=name, global_phase=global_phase, metadata=metadata)
    custom_operations = _read_custom_operations(file_obj, version, vectors)
    for _instruction in range(num_instructions):
        _read_instruction(file_obj, circ, out_registers, custom_operations, version, vectors, use_symengine)
    if version >= 5:
        circ.calibrations = _read_calibrations(file_obj, version, vectors, metadata_deserializer)
    for vec_name, (vector, initialized_params) in vectors.items():
        if len(initialized_params) != len(vector):
            warnings.warn(f"The ParameterVector: '{vec_name}' is not fully identical to its pre-serialization state. Elements {', '.join([str(x) for x in set(range(len(vector))) - initialized_params])} in the ParameterVector will be not equal to the pre-serialized ParameterVector as they weren't used in the circuit: {circ.name}", UserWarning)
    if version >= 8:
        if version >= 10:
            _read_layout_v2(file_obj, circ)
        else:
            _read_layout(file_obj, circ)
    return circ