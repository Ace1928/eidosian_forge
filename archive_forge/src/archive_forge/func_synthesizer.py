from __future__ import annotations
from typing import Union, Callable, Optional, TYPE_CHECKING
from qiskit.circuit import QuantumCircuit
from qiskit.utils import optionals as _optionals
def synthesizer(boolean_expression):
    from tweedledum.synthesis import pkrm_synth
    from qiskit.circuit.classicalfunction.utils import tweedledum2qiskit
    truth_table = boolean_expression._tweedledum_bool_expression.truth_table(output_bit=0)
    tweedledum_circuit = pkrm_synth(truth_table, {'pkrm_synth': {'phase_esop': True}})
    return tweedledum2qiskit(tweedledum_circuit)