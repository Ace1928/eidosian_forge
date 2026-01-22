from __future__ import unicode_literals
from .nodes import FilterNode, filter_operator
from ._utils import escape_chars
@filter_operator()
def filter_multi_output(stream_spec, filter_name, *args, **kwargs):
    """Apply custom filter with one or more outputs.

    This is the same as ``filter`` except that the filter can produce more than one output.

    To reference an output stream, use either the ``.stream`` operator or bracket shorthand:

    Example:

        ```
        split = ffmpeg.input('in.mp4').filter_multi_output('split')
        split0 = split.stream(0)
        split1 = split[1]
        ffmpeg.concat(split0, split1).output('out.mp4').run()
        ```
    """
    return FilterNode(stream_spec, filter_name, args=args, kwargs=kwargs, max_inputs=None)