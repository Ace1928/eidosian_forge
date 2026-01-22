from cirq import circuits
from cirq.contrib.qasm_import._parser import QasmParser
def circuit_from_qasm(qasm: str) -> circuits.Circuit:
    """Parses an OpenQASM string to `cirq.Circuit`.

    Args:
        qasm: The OpenQASM string

    Returns:
        The parsed circuit
    """
    return QasmParser().parse(qasm).circuit