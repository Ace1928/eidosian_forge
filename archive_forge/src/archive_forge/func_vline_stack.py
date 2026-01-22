from __future__ import annotations
from typing import TYPE_CHECKING
import logging # isort:skip
import numpy as np
from ..core.enums import HorizontalLocation, MarkerType, VerticalLocation
from ..core.properties import (
from ..models import (
from ..models.dom import Template
from ..models.tools import (
from ..transform import linear_cmap
from ..util.options import Options
from ._graph import get_graph_kwargs
from ._plot import get_range, get_scale, process_axis_and_grid
from ._stack import double_stack, single_stack
from ._tools import process_active_tools, process_tools_arg
from .contour import ContourRenderer, from_contour
from .glyph_api import _MARKER_SHORTCUTS, GlyphAPI
def vline_stack(self, stackers, **kw):
    """ Generate multiple ``Line`` renderers for lines stacked vertically.

        Args:
            stackers (seq[str]) : a list of data source field names to stack
                successively for ``y`` line coordinates.

        Additionally, the ``name`` of the renderer will be set to
        the value of each successive stacker (this is useful with the
        special hover variable ``$name``)

        Any additional keyword arguments are passed to each call to ``line``.
        If a keyword value is a list or tuple, then each call will get one
        value from the sequence.

        Returns:
            list[GlyphRenderer]

        Examples:

            Assuming a ``ColumnDataSource`` named ``source`` with columns
            *2016* and *2017*, then the following call to ``vline_stack`` with
            stackers for the y-coordinates will will create two ``Line``
            renderers that stack:

            .. code-block:: python

                p.vline_stack(['2016', '2017'], x='x', color=['blue', 'red'], source=source)

            This is equivalent to the following two separate calls:

            .. code-block:: python

                p.line(y=stack('2016'),         x='x', color='blue', source=source, name='2016')
                p.line(y=stack('2016', '2017'), x='x', color='red',  source=source, name='2017')

        """
    return self._line_stack(y=stackers, **kw)