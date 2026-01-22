from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
from ..models import glyphs
from ..util.deprecation import deprecated
from ._decorators import glyph_method, marker_method
@glyph_method(glyphs.MultiPolygons)
def multi_polygons(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
    """
.. note::
    For this glyph, the data is not simply an array of scalars, it is a
    nested array.

Examples:

    .. code-block:: python

        from bokeh.plotting import figure, output_file, show

        p = figure(width=300, height=300)
        p.multi_polygons(xs=[[[[1, 1, 2, 2]]], [[[1, 1, 3], [1.5, 1.5, 2]]]],
                        ys=[[[[4, 3, 3, 4]]], [[[1, 3, 1], [1.5, 2, 1.5]]]],
                        color=['red', 'green'])
        show(p)

"""