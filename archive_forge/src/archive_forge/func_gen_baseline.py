from typing import Dict, Any, List
from qiskit.visualization.pulse_v2 import drawings, types, device_info
def gen_baseline(data: types.ChartAxis, formatter: Dict[str, Any], device: device_info.DrawerBackendInfo) -> List[drawings.LineData]:
    """Generate the baseline associated with the chart.

    Stylesheets:
        - The `baseline` style is applied.

    Args:
        data: Chart axis data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Returns:
        List of `LineData` drawings.
    """
    style = {'alpha': formatter['alpha.baseline'], 'zorder': formatter['layer.baseline'], 'linewidth': formatter['line_width.baseline'], 'linestyle': formatter['line_style.baseline'], 'color': formatter['color.baseline']}
    baseline = drawings.LineData(data_type=types.LineType.BASELINE, channels=data.channels, xvals=[types.AbstractCoordinate.LEFT, types.AbstractCoordinate.RIGHT], yvals=[0, 0], ignore_scaling=True, styles=style)
    return [baseline]