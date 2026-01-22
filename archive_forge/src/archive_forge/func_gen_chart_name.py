from typing import Dict, Any, List
from qiskit.visualization.pulse_v2 import drawings, types, device_info
def gen_chart_name(data: types.ChartAxis, formatter: Dict[str, Any], device: device_info.DrawerBackendInfo) -> List[drawings.TextData]:
    """Generate the name of chart.

    Stylesheets:
        - The `axis_label` style is applied.

    Args:
        data: Chart axis data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Returns:
        List of `TextData` drawings.
    """
    style = {'zorder': formatter['layer.axis_label'], 'color': formatter['color.axis_label'], 'size': formatter['text_size.axis_label'], 'va': 'center', 'ha': 'right'}
    text = drawings.TextData(data_type=types.LabelType.CH_NAME, channels=data.channels, xvals=[types.AbstractCoordinate.LEFT], yvals=[0], text=data.name, ignore_scaling=True, styles=style)
    return [text]