from fractions import Fraction
from typing import Dict, Any, List, Tuple
import numpy as np
from qiskit.pulse import instructions
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import drawings, types, device_info
def gen_frame_symbol(data: types.PulseInstruction, formatter: Dict[str, Any], device: device_info.DrawerBackendInfo) -> List[drawings.TextData]:
    """Generate a frame change symbol with instruction meta data from provided frame instruction.

    Stylesheets:
        - The `frame_change` style is applied.
        - The symbol type in unicode is specified in `formatter.unicode_symbol.frame_change`.
        - The symbol type in latex is specified in `formatter.latex_symbol.frame_change`.

    Args:
        data: Frame change instruction data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Returns:
        List of `TextData` drawings.
    """
    if data.frame.phase == 0 and data.frame.freq == 0:
        return []
    style = {'zorder': formatter['layer.frame_change'], 'color': formatter['color.frame_change'], 'size': formatter['text_size.frame_change'], 'va': 'center', 'ha': 'center'}
    program = []
    for inst in data.inst:
        if isinstance(inst, (instructions.SetFrequency, instructions.ShiftFrequency)):
            try:
                program.append(f'{inst.__class__.__name__}({inst.frequency:.2e} Hz)')
            except TypeError:
                program.append(f'{inst.__class__.__name__}({inst.frequency})')
        elif isinstance(inst, (instructions.SetPhase, instructions.ShiftPhase)):
            try:
                program.append(f'{inst.__class__.__name__}({inst.phase:.2f} rad.)')
            except TypeError:
                program.append(f'{inst.__class__.__name__}({inst.phase})')
    meta = {'total phase change': data.frame.phase, 'total frequency change': data.frame.freq, 'program': ', '.join(program), 't0 (cycle time)': data.t0, 't0 (sec)': data.t0 * data.dt if data.dt else 'N/A'}
    text = drawings.TextData(data_type=types.SymbolType.FRAME, channels=data.inst[0].channel, xvals=[data.t0], yvals=[0], text=formatter['unicode_symbol.frame_change'], latex=formatter['latex_symbol.frame_change'], ignore_scaling=True, meta=meta, styles=style)
    return [text]