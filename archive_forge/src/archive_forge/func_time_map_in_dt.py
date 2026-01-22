from typing import List, Tuple
import numpy as np
from qiskit import circuit
from qiskit.visualization.timeline import types
def time_map_in_dt(time_window: Tuple[int, int]) -> types.HorizontalAxis:
    """Layout function for the horizontal axis formatting.

    Generate equispaced 6 horizontal axis ticks.

    Args:
        time_window: Left and right edge of this graph.

    Returns:
        Axis formatter object.
    """
    t0, t1 = time_window
    axis_loc = np.linspace(max(t0, 0), t1, 6)
    axis_label = axis_loc.copy()
    label = 'System cycle time (dt)'
    formatted_label = [f'{val:.0f}' for val in axis_label]
    return types.HorizontalAxis(window=(t0, t1), axis_map=dict(zip(axis_loc, formatted_label)), label=label)