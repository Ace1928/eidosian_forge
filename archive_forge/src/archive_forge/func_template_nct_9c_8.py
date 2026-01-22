from qiskit.circuit.quantumcircuit import QuantumCircuit
def template_nct_9c_8():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(3)
    qc.ccx(0, 2, 1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.ccx(0, 2, 1)
    qc.x(2)
    qc.ccx(0, 2, 1)
    qc.cx(1, 2)
    qc.x(2)
    qc.ccx(0, 2, 1)
    return qc