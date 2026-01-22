from qiskit.circuit.quantumcircuit import QuantumCircuit
def template_nct_2a_1():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.x(0)
    return qc