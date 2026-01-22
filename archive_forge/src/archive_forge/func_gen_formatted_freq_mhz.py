from fractions import Fraction
from typing import Dict, Any, List, Tuple
import numpy as np
from qiskit.pulse import instructions
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import drawings, types, device_info
def gen_formatted_freq_mhz(data: types.PulseInstruction, formatter: Dict[str, Any], device: device_info.DrawerBackendInfo) -> List[drawings.TextData]:
    """Generate the formatted frequency change label from provided frame instruction.

    Frequency change is expressed in units of MHz.

    For example:
        - A value 1,234,567 is converted into `\\Delta f = 1.23 MHz`

    Stylesheets:
        - The `frame_change` style is applied.
        - The `annotate` style is applied for font size.

    Args:
        data: Frame change instruction data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Returns:
        List of `TextData` drawings.
    """
    _unit = 'MHz'
    style = {'zorder': formatter['layer.frame_change'], 'color': formatter['color.frame_change'], 'size': formatter['text_size.annotate'], 'va': 'center', 'ha': 'center'}
    plain_freq, latex_freq = _freq_to_text(formatter=formatter, freq=data.frame.freq, unit=_unit)
    text = drawings.TextData(data_type=types.LabelType.FRAME, channels=data.inst[0].channel, xvals=[data.t0], yvals=[formatter['label_offset.frame_change']], text=f'Δf = {plain_freq}', latex=f'\\Delta f = {latex_freq}', ignore_scaling=True, styles=style)
    return [text]