from qiskit.transpiler.target import Target, InstructionProperties
from qiskit.providers.backend import QubitProperties
from qiskit.utils.units import apply_prefix
from qiskit.circuit.library.standard_gates import IGate, SXGate, XGate, CXGate, RZGate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.gate import Gate
from qiskit.circuit.delay import Delay
from qiskit.circuit.measure import Measure
from qiskit.circuit.reset import Reset
from qiskit.providers.models.pulsedefaults import PulseDefaults
def qubit_props_from_props(properties: dict) -> list:
    """Returns a dictionary of `qiskit.providers.backend.QubitProperties` using
    a backend properties dictionary created by loading props.json payload.
    """
    qubit_props = []
    for qubit in properties['qubits']:
        qubit_properties = {}
        for prop_dict in qubit:
            if prop_dict['name'] == 'T1':
                qubit_properties['t1'] = apply_prefix(prop_dict['value'], prop_dict['unit'])
            elif prop_dict['name'] == 'T2':
                qubit_properties['t2'] = apply_prefix(prop_dict['value'], prop_dict['unit'])
            elif prop_dict['name'] == 'frequency':
                qubit_properties['frequency'] = apply_prefix(prop_dict['value'], prop_dict['unit'])
        qubit_props.append(QubitProperties(**qubit_properties))
    return qubit_props