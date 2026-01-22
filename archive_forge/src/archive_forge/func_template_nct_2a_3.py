from qiskit.circuit.quantumcircuit import QuantumCircuit
def template_nct_2a_3():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    qc.ccx(0, 1, 2)
    return qc