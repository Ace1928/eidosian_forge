from traitlets import Unicode, Instance, Bool
from ipywidgets import DOMWidget, register, widget_serialization
from .interacts import PanZoom
from .figure import Figure
from ._version import __frontend_version__
Default toolbar for bqplot figures.

    The default toolbar provides three buttons:

    - A *Panzoom* toggle button which enables panning and zooming the figure.
    - A *Save* button to save the figure as a png image.
    - A *Reset* button, which resets the figure position to its original
      state.

    When the *Panzoom* button is toggled to True for the first time, a new
    instance of ``PanZoom`` widget is created.
    The created ``PanZoom`` widget uses the scales of all the marks that are on
    the figure at this point.
    When the *PanZoom* widget is toggled to False, the figure retrieves its
    previous interaction.
    When the *Reset* button is pressed, the ``PanZoom`` widget is deleted and
    the figure scales reset to their initial state. We are back to the case
    where the PanZoom widget has never been set.

    If new marks are added to the figure after the panzoom button is toggled,
    and these use new scales, those scales will not be panned or zoomed,
    unless the reset button is clicked.

    Attributes
    ----------
    figure: instance of Figure
        The figure to which the toolbar will apply.
    