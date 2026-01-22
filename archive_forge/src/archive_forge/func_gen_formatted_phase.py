from fractions import Fraction
from typing import Dict, Any, List, Tuple
import numpy as np
from qiskit.pulse import instructions
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import drawings, types, device_info
def gen_formatted_phase(data: types.PulseInstruction, formatter: Dict[str, Any], device: device_info.DrawerBackendInfo) -> List[drawings.TextData]:
    """Generate the formatted virtual Z rotation label from provided frame instruction.

    Rotation angle is expressed in units of pi.
    If the denominator of fraction is larger than 10, the angle is expressed in units of radian.

    For example:
        - A value -3.14 is converted into `VZ(\\pi)`
        - A value 1.57 is converted into `VZ(-\\frac{\\pi}{2})`
        - A value 0.123 is converted into `VZ(-0.123 rad.)`

    Stylesheets:
        - The `frame_change` style is applied.
        - The `annotate` style is applied for font size.

    Notes:
        The phase operand of `PhaseShift` instruction has opposite sign to the Z gate definition.
        Thus the sign of rotation angle is inverted.

    Args:
        data: Frame change instruction data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Returns:
        List of `TextData` drawings.
    """
    _max_denom = 10
    style = {'zorder': formatter['layer.frame_change'], 'color': formatter['color.frame_change'], 'size': formatter['text_size.annotate'], 'va': 'center', 'ha': 'center'}
    plain_phase, latex_phase = _phase_to_text(formatter=formatter, phase=data.frame.phase, max_denom=_max_denom, flip=True)
    text = drawings.TextData(data_type=types.LabelType.FRAME, channels=data.inst[0].channel, xvals=[data.t0], yvals=[formatter['label_offset.frame_change']], text=f'VZ({plain_phase})', latex=f'{{\\rm VZ}}({latex_phase})', ignore_scaling=True, styles=style)
    return [text]