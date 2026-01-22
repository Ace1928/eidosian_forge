from contextlib import ExitStack
import inspect
import itertools
import logging
from numbers import Integral
import threading
import numpy as np
import matplotlib as mpl
from matplotlib import _blocking_input, backend_bases, _docstring, projections
from matplotlib.artist import (
from matplotlib.backend_bases import (
import matplotlib._api as _api
import matplotlib.cbook as cbook
import matplotlib.colorbar as cbar
import matplotlib.image as mimage
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.layout_engine import (
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import (Affine2D, Bbox, BboxTransformTo,
def ginput(self, n=1, timeout=30, show_clicks=True, mouse_add=MouseButton.LEFT, mouse_pop=MouseButton.RIGHT, mouse_stop=MouseButton.MIDDLE):
    """
        Blocking call to interact with a figure.

        Wait until the user clicks *n* times on the figure, and return the
        coordinates of each click in a list.

        There are three possible interactions:

        - Add a point.
        - Remove the most recently added point.
        - Stop the interaction and return the points added so far.

        The actions are assigned to mouse buttons via the arguments
        *mouse_add*, *mouse_pop* and *mouse_stop*.

        Parameters
        ----------
        n : int, default: 1
            Number of mouse clicks to accumulate. If negative, accumulate
            clicks until the input is terminated manually.
        timeout : float, default: 30 seconds
            Number of seconds to wait before timing out. If zero or negative
            will never time out.
        show_clicks : bool, default: True
            If True, show a red cross at the location of each click.
        mouse_add : `.MouseButton` or None, default: `.MouseButton.LEFT`
            Mouse button used to add points.
        mouse_pop : `.MouseButton` or None, default: `.MouseButton.RIGHT`
            Mouse button used to remove the most recently added point.
        mouse_stop : `.MouseButton` or None, default: `.MouseButton.MIDDLE`
            Mouse button used to stop input.

        Returns
        -------
        list of tuples
            A list of the clicked (x, y) coordinates.

        Notes
        -----
        The keyboard can also be used to select points in case your mouse
        does not have one or more of the buttons.  The delete and backspace
        keys act like right-clicking (i.e., remove last point), the enter key
        terminates input and any other key (not already used by the window
        manager) selects a point.
        """
    clicks = []
    marks = []

    def handler(event):
        is_button = event.name == 'button_press_event'
        is_key = event.name == 'key_press_event'
        if is_button and event.button == mouse_stop or (is_key and event.key in ['escape', 'enter']):
            self.canvas.stop_event_loop()
        elif is_button and event.button == mouse_pop or (is_key and event.key in ['backspace', 'delete']):
            if clicks:
                clicks.pop()
                if show_clicks:
                    marks.pop().remove()
                    self.canvas.draw()
        elif is_button and event.button == mouse_add or (is_key and event.key is not None):
            if event.inaxes:
                clicks.append((event.xdata, event.ydata))
                _log.info('input %i: %f, %f', len(clicks), event.xdata, event.ydata)
                if show_clicks:
                    line = mpl.lines.Line2D([event.xdata], [event.ydata], marker='+', color='r')
                    event.inaxes.add_line(line)
                    marks.append(line)
                    self.canvas.draw()
        if len(clicks) == n and n > 0:
            self.canvas.stop_event_loop()
    _blocking_input.blocking_input_loop(self, ['button_press_event', 'key_press_event'], timeout, handler)
    for mark in marks:
        mark.remove()
    self.canvas.draw()
    return clicks