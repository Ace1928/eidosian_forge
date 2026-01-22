from bokeh.core.properties import (
from bokeh.events import ModelEvent
from bokeh.models import ColumnDataSource, LayoutDOM
from bokeh.models.widgets.tables import TableColumn
from ..config import config
from ..io.resources import bundled_files
from ..util import classproperty
from .layout import HTMLBox
 Selection Event

        Parameters
        ----------
        model : ModelEvent
            An event send when a selection is changed on the frontend.
        indices : list[int]
            A list of changed indices selected/deselected rows.
        selected : bool
            If true the rows were selected, if false they were deselected.
        flush : bool
            Whether the current selection should be emptied before adding the new indices.
        