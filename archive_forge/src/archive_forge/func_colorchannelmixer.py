from __future__ import unicode_literals
from .nodes import FilterNode, filter_operator
from ._utils import escape_chars
@filter_operator()
def colorchannelmixer(stream, *args, **kwargs):
    """Adjust video input frames by re-mixing color channels.

    Official documentation: `colorchannelmixer <https://ffmpeg.org/ffmpeg-filters.html#colorchannelmixer>`__
    """
    return FilterNode(stream, colorchannelmixer.__name__, kwargs=kwargs).stream()