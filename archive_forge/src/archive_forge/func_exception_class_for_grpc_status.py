from __future__ import absolute_import
from __future__ import unicode_literals
import http.client
from typing import Dict
from typing import Union
import warnings
from google.rpc import error_details_pb2
def exception_class_for_grpc_status(status_code):
    """Return the exception class for a specific :class:`grpc.StatusCode`.

    Args:
        status_code (grpc.StatusCode): The gRPC status code.

    Returns:
        :func:`type`: the appropriate subclass of :class:`GoogleAPICallError`.
    """
    return _GRPC_CODE_TO_EXCEPTION.get(status_code, GoogleAPICallError)