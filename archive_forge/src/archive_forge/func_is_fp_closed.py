from __future__ import absolute_import
from email.errors import MultipartInvariantViolationDefect, StartBoundaryNotFoundDefect
from ..exceptions import HeaderParsingError
from ..packages.six.moves import http_client as httplib
def is_fp_closed(obj):
    """
    Checks whether a given file-like object is closed.

    :param obj:
        The file-like object to check.
    """
    try:
        return obj.isclosed()
    except AttributeError:
        pass
    try:
        return obj.closed
    except AttributeError:
        pass
    try:
        return obj.fp is None
    except AttributeError:
        pass
    raise ValueError('Unable to determine whether fp is closed.')