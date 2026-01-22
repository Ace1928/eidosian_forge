import warnings
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.utils.units import apply_prefix
def convert_durations_to_dt(qc: QuantumCircuit, dt_in_sec: float, inplace=True):
    """Convert all the durations in SI (seconds) into those in dt.

    Returns a new circuit if `inplace=False`.

    Parameters:
        qc (QuantumCircuit): Duration of dt in seconds used for conversion.
        dt_in_sec (float): Duration of dt in seconds used for conversion.
        inplace (bool): All durations are converted inplace or return new circuit.

    Returns:
        QuantumCircuit: Converted circuit if `inplace = False`, otherwise None.

    Raises:
        CircuitError: if fail to convert durations.
    """
    if inplace:
        circ = qc
    else:
        circ = qc.copy()
    for instruction in circ.data:
        operation = instruction.operation
        if operation.unit == 'dt' or operation.duration is None:
            continue
        if not operation.unit.endswith('s'):
            raise CircuitError(f"Invalid time unit: '{operation.unit}'")
        duration = operation.duration
        if operation.unit != 's':
            duration = apply_prefix(duration, operation.unit)
        operation.duration = duration_in_dt(duration, dt_in_sec)
        operation.unit = 'dt'
    if circ.duration is not None and circ.unit != 'dt':
        if not circ.unit.endswith('s'):
            raise CircuitError(f"Invalid time unit: '{circ.unit}'")
        duration = circ.duration
        if circ.unit != 's':
            duration = apply_prefix(duration, circ.unit)
        circ.duration = duration_in_dt(duration, dt_in_sec)
        circ.unit = 'dt'
    if not inplace:
        return circ
    else:
        return None