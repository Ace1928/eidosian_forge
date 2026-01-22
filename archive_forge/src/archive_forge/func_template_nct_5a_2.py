from qiskit.circuit.quantumcircuit import QuantumCircuit
def template_nct_5a_2():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    qc.x(1)
    qc.ccx(0, 1, 2)
    qc.x(1)
    qc.cx(0, 2)
    return qc