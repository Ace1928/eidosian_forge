from __future__ import unicode_literals
from .nodes import FilterNode, filter_operator
from ._utils import escape_chars
@filter_operator()
def setpts(stream, expr):
    """Change the PTS (presentation timestamp) of the input frames.

    Args:
        expr: The expression which is evaluated for each frame to construct its timestamp.

    Official documentation: `setpts, asetpts <https://ffmpeg.org/ffmpeg-filters.html#setpts_002c-asetpts>`__
    """
    return FilterNode(stream, setpts.__name__, args=[expr]).stream()