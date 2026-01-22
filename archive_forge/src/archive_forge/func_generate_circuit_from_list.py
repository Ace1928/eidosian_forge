import json
from typing import Any, cast, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import networkx as nx
import numpy as np
import cirq
from cirq_aqt import aqt_device_metadata
def generate_circuit_from_list(self, json_string: str):
    """Generates a list of cirq operations from a json string.

        The default behavior is to add a measurement to any qubit at the end
        of the circuit as there are no measurements defined in the AQT API.

        Args:
            json_string: json that specifies the sequence.
        """
    self.circuit = cirq.Circuit()
    json_obj = json.loads(json_string)
    gate: Union[cirq.PhasedXPowGate, cirq.EigenGate]
    for circuit_list in json_obj:
        op_str = circuit_list[0]
        if op_str == 'R':
            gate = cast(cirq.PhasedXPowGate, gate_dict[op_str])
            theta = circuit_list[1]
            phi = circuit_list[2]
            qubits = [self.qubit_list[i] for i in circuit_list[3]]
            self.circuit.append(gate(phase_exponent=phi, exponent=theta).on(*qubits))
        else:
            gate = cast(cirq.EigenGate, gate_dict[op_str])
            angle = circuit_list[1]
            qubits = [self.qubit_list[i] for i in circuit_list[2]]
            self.circuit.append(gate.on(*qubits) ** angle)
    self.circuit.append(cirq.measure(*[qubit for qubit in self.qubit_list], key='m'))