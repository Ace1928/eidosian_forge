from __future__ import unicode_literals
from .nodes import FilterNode, filter_operator
from ._utils import escape_chars
@filter_operator()
def zoompan(stream, **kwargs):
    """Apply Zoom & Pan effect.

    Args:
        zoom: Set the zoom expression. Default is 1.
        x: Set the x expression. Default is 0.
        y: Set the y expression. Default is 0.
        d: Set the duration expression in number of frames. This sets for how many number of frames effect will last
            for single input image.
        s: Set the output image size, default is ``hd720``.
        fps: Set the output frame rate, default is 25.
        z: Alias for ``zoom``.

    Official documentation: `zoompan <https://ffmpeg.org/ffmpeg-filters.html#zoompan>`__
    """
    return FilterNode(stream, zoompan.__name__, kwargs=kwargs).stream()