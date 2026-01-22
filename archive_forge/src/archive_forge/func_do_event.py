from unittest import mock
import matplotlib.pyplot as plt
def do_event(tool, etype, button=1, xdata=0, ydata=0, key=None, step=1):
    """
    Trigger an event on the given tool.

    Parameters
    ----------
    tool : matplotlib.widgets.AxesWidget
    etype : str
        The event to trigger.
    xdata : float
        x coord of mouse in data coords.
    ydata : float
        y coord of mouse in data coords.
    button : None or `MouseButton` or {'up', 'down'}
        The mouse button pressed in this event (see also `.MouseEvent`).
    key : None or str
        The key pressed when the mouse event triggered (see also `.KeyEvent`).
    step : int
        Number of scroll steps (positive for 'up', negative for 'down').
    """
    event = mock_event(tool.ax, button, xdata, ydata, key, step)
    func = getattr(tool, etype)
    func(event)