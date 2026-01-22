import copy
import functools
import sys
import warnings
import grpc
from grpc._cython import cygrpc as _cygrpc
def wrap_server_method_handler(wrapper, handler):
    """Wraps the server method handler function.

    The server implementation requires all server handlers being wrapped as
    RpcMethodHandler objects. This helper function ease the pain of writing
    server handler wrappers.

    Args:
        wrapper: A wrapper function that takes in a method handler behavior
          (the actual function) and returns a wrapped function.
        handler: A RpcMethodHandler object to be wrapped.

    Returns:
        A newly created RpcMethodHandler.
    """
    if not handler:
        return None
    if not handler.request_streaming:
        if not handler.response_streaming:
            return handler._replace(unary_unary=wrapper(handler.unary_unary))
        else:
            return handler._replace(unary_stream=wrapper(handler.unary_stream))
    elif not handler.response_streaming:
        return handler._replace(stream_unary=wrapper(handler.stream_unary))
    else:
        return handler._replace(stream_stream=wrapper(handler.stream_stream))