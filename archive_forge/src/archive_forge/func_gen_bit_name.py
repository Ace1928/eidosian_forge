import warnings
from typing import List, Union, Dict, Any, Optional
from qiskit.circuit import Qubit, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.visualization.timeline import types, drawings
def gen_bit_name(bit: types.Bits, formatter: Dict[str, Any], program: Optional[QuantumCircuit]=None) -> List[drawings.TextData]:
    """Generate bit label.

    Stylesheet:
        - `bit_name` style is applied.

    Args:
        bit: Bit object associated to this drawing.
        formatter: Dictionary of stylesheet settings.
        program: Optional program that the bits are a part of.

    Returns:
        List of `TextData` drawings.
    """
    styles = {'zorder': formatter['layer.bit_name'], 'color': formatter['color.bit_name'], 'size': formatter['text_size.bit_name'], 'va': 'center', 'ha': 'right'}
    if program is None:
        warnings.warn("bits cannot be accurately named without passing a 'program'", stacklevel=2)
        label_plain = 'q' if isinstance(bit, Qubit) else 'c'
        label_latex = f'{{\\rm {label_plain}}}'
    else:
        loc = program.find_bit(bit)
        if loc.registers:
            label_plain = loc.registers[-1][0].name
            label_latex = f'{{\\rm {loc.registers[-1][0].prefix}}}_{{{loc.registers[-1][1]}}}'
        else:
            label_plain = 'q' if isinstance(bit, Qubit) else 'c'
            label_latex = f'{{\\rm {label_plain}}}_{{{loc.index}}}'
    drawing = drawings.TextData(data_type=types.LabelType.BIT_NAME, xval=types.AbstractCoordinate.LEFT, yval=0, bit=bit, text=label_plain, latex=label_latex, styles=styles)
    return [drawing]