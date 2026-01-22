from typing import Optional, Tuple
from cirq import ops, protocols
def convert_text_diagram_info_to_qcircuit_diagram_info(info: protocols.CircuitDiagramInfo) -> protocols.CircuitDiagramInfo:
    labels = [escape_text_for_latex(e) for e in info.wire_symbols]
    if info.exponent != 1:
        labels[0] += '^{' + str(info.exponent) + '}'
    symbols = tuple(('\\gate{' + l + '}' for l in labels))
    return protocols.CircuitDiagramInfo(symbols)