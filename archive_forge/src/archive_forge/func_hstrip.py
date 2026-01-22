from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
from ..models import glyphs
from ..util.deprecation import deprecated
from ._decorators import glyph_method, marker_method
@glyph_method(glyphs.HStrip)
def hstrip(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
    """
Examples:

    .. code-block:: python

        from bokeh.plotting import figure, output_file, show

        plot = figure(width=300, height=300, x_range=(0, 1))
        plot.hstrip(y0=[1, 2, 5], y1=[3, 4, 8], color="#CAB2D6")

        show(plot)

"""