from fractions import Fraction
from typing import Dict, Any, List, Tuple
import numpy as np
from qiskit.pulse import instructions
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import drawings, types, device_info
def gen_formatted_frame_values(data: types.PulseInstruction, formatter: Dict[str, Any], device: device_info.DrawerBackendInfo) -> List[drawings.TextData]:
    """Generate the formatted virtual Z rotation label and the frequency change label
    from provided frame instruction.

    Phase value is placed on top of the symbol, and frequency value is placed below the symbol.
    See :py:func:`gen_formatted_phase` and :py:func:`gen_formatted_freq_mhz` for details.

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
    texts = []
    _max_denom = 10
    _unit = 'MHz'
    style = {'zorder': formatter['layer.frame_change'], 'color': formatter['color.frame_change'], 'size': formatter['text_size.annotate'], 'ha': 'center'}
    if data.frame.phase != 0:
        plain_phase, latex_phase = _phase_to_text(formatter=formatter, phase=data.frame.phase, max_denom=_max_denom, flip=True)
        phase_style = {'va': 'center'}
        phase_style.update(style)
        phase = drawings.TextData(data_type=types.LabelType.FRAME, channels=data.inst[0].channel, xvals=[data.t0], yvals=[formatter['label_offset.frame_change']], text=f'VZ({plain_phase})', latex=f'{{\\rm VZ}}({latex_phase})', ignore_scaling=True, styles=phase_style)
        texts.append(phase)
    if data.frame.freq != 0:
        plain_freq, latex_freq = _freq_to_text(formatter=formatter, freq=data.frame.freq, unit=_unit)
        freq_style = {'va': 'center'}
        freq_style.update(style)
        freq = drawings.TextData(data_type=types.LabelType.FRAME, channels=data.inst[0].channel, xvals=[data.t0], yvals=[2 * formatter['label_offset.frame_change']], text=f'Δf = {plain_freq}', latex=f'\\Delta f = {latex_freq}', ignore_scaling=True, styles=freq_style)
        texts.append(freq)
    return texts