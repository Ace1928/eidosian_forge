from qiskit.circuit.quantumcircuit import QuantumCircuit
def template_nct_7e_1():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.ccx(0, 2, 1)
    qc.x(2)
    qc.cx(0, 2)
    qc.ccx(0, 2, 1)
    qc.x(0)
    qc.cx(0, 2)
    return qc