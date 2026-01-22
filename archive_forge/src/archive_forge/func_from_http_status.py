from __future__ import absolute_import
from __future__ import unicode_literals
import http.client
from typing import Dict
from typing import Union
import warnings
from google.rpc import error_details_pb2
def from_http_status(status_code, message, **kwargs):
    """Create a :class:`GoogleAPICallError` from an HTTP status code.

    Args:
        status_code (int): The HTTP status code.
        message (str): The exception message.
        kwargs: Additional arguments passed to the :class:`GoogleAPICallError`
            constructor.

    Returns:
        GoogleAPICallError: An instance of the appropriate subclass of
            :class:`GoogleAPICallError`.
    """
    error_class = exception_class_for_http_status(status_code)
    error = error_class(message, **kwargs)
    if error.code is None:
        error.code = status_code
    return error