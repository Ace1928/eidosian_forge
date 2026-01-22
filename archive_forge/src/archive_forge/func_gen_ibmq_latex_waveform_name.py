from __future__ import annotations
import re
from fractions import Fraction
from typing import Any
import numpy as np
from qiskit import pulse, circuit
from qiskit.pulse import instructions, library
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import drawings, types, device_info
def gen_ibmq_latex_waveform_name(data: types.PulseInstruction, formatter: dict[str, Any], device: device_info.DrawerBackendInfo) -> list[drawings.TextData]:
    """Generate the formatted instruction name associated with the waveform.

    Channel name and ID string are removed and the rotation angle is expressed in units of pi.
    The controlled rotation angle associated with the CR pulse name is divided by 2.

    Note that in many scientific articles the controlled rotation angle implies
    the actual rotation angle, but in IQX backend the rotation angle represents
    the difference between rotation angles with different control qubit states.

    For example:
        - 'X90p_d0_abcdefg' is converted into 'X(\\frac{\\pi}{2})'
        - 'CR90p_u0_abcdefg` is converted into 'CR(\\frac{\\pi}{4})'

    Stylesheets:
        - The `annotate` style is applied.

    Notes:
        This generator can convert pulse names used in the IQX backends.
        If pulses are provided by the third party providers or the user defined,
        the generator output may be the as-is pulse name.

    Args:
        data: Waveform instruction data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Returns:
        List of `TextData` drawings.
    """
    if data.is_opaque:
        return []
    style = {'zorder': formatter['layer.annotate'], 'color': formatter['color.annotate'], 'size': formatter['text_size.annotate'], 'va': 'center', 'ha': 'center'}
    if isinstance(data.inst, pulse.instructions.Acquire):
        systematic_name = 'Acquire'
        latex_name = None
    elif isinstance(data.inst, instructions.Delay):
        systematic_name = data.inst.name or 'Delay'
        latex_name = None
    else:
        pulse_data = data.inst.pulse
        if pulse_data.name:
            systematic_name = pulse_data.name
        elif isinstance(pulse_data, library.SymbolicPulse):
            systematic_name = pulse_data.pulse_type
        else:
            systematic_name = 'Waveform'
        template = '(?P<op>[A-Z]+)(?P<angle>[0-9]+)?(?P<sign>[pm])_(?P<ch>[dum])[0-9]+'
        match_result = re.match(template, systematic_name)
        if match_result is not None:
            match_dict = match_result.groupdict()
            sign = '' if match_dict['sign'] == 'p' else '-'
            if match_dict['op'] == 'CR':
                if match_dict['ch'] == 'u':
                    op_name = '{\\rm CR}'
                else:
                    op_name = '\\overline{\\rm CR}'
                angle_val = match_dict['angle']
                frac = Fraction(int(int(angle_val) / 2), 180)
                if frac.numerator == 1:
                    angle = f'\\pi/{frac.denominator:d}'
                else:
                    angle = '{num:d}/{denom:d} \\pi'.format(num=frac.numerator, denom=frac.denominator)
            else:
                op_name = '{{\\rm {}}}'.format(match_dict['op'])
                angle_val = match_dict['angle']
                if angle_val is None:
                    angle = '\\pi'
                else:
                    frac = Fraction(int(angle_val), 180)
                    if frac.numerator == 1:
                        angle = f'\\pi/{frac.denominator:d}'
                    else:
                        angle = '{num:d}/{denom:d} \\pi'.format(num=frac.numerator, denom=frac.denominator)
            latex_name = f'{op_name}({sign}{angle})'
        else:
            latex_name = None
    text = drawings.TextData(data_type=types.LabelType.PULSE_NAME, channels=data.inst.channel, xvals=[data.t0 + 0.5 * data.inst.duration], yvals=[-formatter['label_offset.pulse_name']], text=systematic_name, latex=latex_name, ignore_scaling=True, styles=style)
    return [text]