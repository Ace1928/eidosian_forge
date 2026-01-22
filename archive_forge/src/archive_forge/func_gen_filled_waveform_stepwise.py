from __future__ import annotations
import re
from fractions import Fraction
from typing import Any
import numpy as np
from qiskit import pulse, circuit
from qiskit.pulse import instructions, library
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import drawings, types, device_info
def gen_filled_waveform_stepwise(data: types.PulseInstruction, formatter: dict[str, Any], device: device_info.DrawerBackendInfo) -> list[drawings.LineData | drawings.BoxData | drawings.TextData]:
    """Generate filled area objects of the real and the imaginary part of waveform envelope.

    The curve of envelope is not interpolated nor smoothed and presented
    as stepwise function at each data point.

    Stylesheets:
        - The `fill_waveform` style is applied.

    Args:
        data: Waveform instruction data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Returns:
        List of `LineData`, `BoxData`, or `TextData` drawings.

    Raises:
        VisualizationError: When the instruction parser returns invalid data format.
    """
    waveform_data = _parse_waveform(data)
    channel = data.inst.channel
    meta = waveform_data.meta
    qind = device.get_qubit_index(channel)
    meta.update({'qubit': qind if qind is not None else 'N/A'})
    if isinstance(waveform_data, types.ParsedInstruction):
        xdata = waveform_data.xvals
        ydata = waveform_data.yvals
        if formatter['control.apply_phase_modulation']:
            ydata = np.asarray(ydata, dtype=complex) * np.exp(1j * data.frame.phase)
        else:
            ydata = np.asarray(ydata, dtype=complex)
        return _draw_shaped_waveform(xdata=xdata, ydata=ydata, meta=meta, channel=channel, formatter=formatter)
    elif isinstance(waveform_data, types.OpaqueShape):
        unbound_params = []
        for pname, pval in data.inst.pulse.parameters.items():
            if isinstance(pval, circuit.ParameterExpression):
                unbound_params.append(pname)
        pulse_data = data.inst.pulse
        if isinstance(pulse_data, library.SymbolicPulse):
            pulse_shape = pulse_data.pulse_type
        else:
            pulse_shape = 'Waveform'
        return _draw_opaque_waveform(init_time=data.t0, duration=waveform_data.duration, pulse_shape=pulse_shape, pnames=unbound_params, meta=meta, channel=channel, formatter=formatter)
    else:
        raise VisualizationError('Invalid data format is provided.')