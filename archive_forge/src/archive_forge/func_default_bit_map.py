from pennylane.ops import Controlled, Conditional
from pennylane.measurements import MeasurementProcess, MidMeasureMP, MeasurementValue
def default_bit_map(tape):
    """Create a dictionary mapping ``MidMeasureMP``'s to indices corresponding to classical
    wires. We only add mid-circuit measurements that are used for classical conditions and for
    collecting statistics to this dictionary.

    Args:
        tape [~.tape.QuantumTape]: the QuantumTape containing operations and measurements

    Returns:
        dict: map from mid-circuit measurements to classical wires."""
    bit_map = {}
    for op in tape:
        if isinstance(op, Conditional):
            for m in op.meas_val.measurements:
                bit_map[m] = None
        if isinstance(op, MeasurementProcess) and op.mv is not None:
            if isinstance(op.mv, MeasurementValue):
                for m in op.mv.measurements:
                    bit_map[m] = None
            else:
                for m in op.mv:
                    bit_map[m.measurements[0]] = None
    cur_cwire = 0
    for op in tape:
        if isinstance(op, MidMeasureMP) and op in bit_map:
            bit_map[op] = cur_cwire
            cur_cwire += 1
    return bit_map