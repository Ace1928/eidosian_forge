from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
from ..models import glyphs
from ..util.deprecation import deprecated
from ._decorators import glyph_method, marker_method
@marker_method()
def star_dot(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
    """
Examples:

    .. code-block:: python

        from bokeh.plotting import figure, output_file, show

        plot = figure(width=300, height=300)
        plot.star_dot(x=[1, 2, 3], y=[1, 2, 3], size=20,
                      color="#386CB0", fill_color=None, line_width=2)

        show(plot)

"""