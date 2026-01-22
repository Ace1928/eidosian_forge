from io import BytesIO
from typing import Callable, Dict, List, Tuple
from .. import errors, osutils, registry
def filtered_input_file(f, filters):
    """Get an input file that converts external to internal content.

    Args:
      f: the original input file
      filters: the stack of filters to apply

    Returns: a file-like object, size
    """
    chunks = [f.read()]
    for filter in filters:
        if filter.reader is not None:
            chunks = filter.reader(chunks)
    text = b''.join(chunks)
    return (BytesIO(text), len(text))