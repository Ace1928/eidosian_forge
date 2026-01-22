import abc
import contextlib
import enum
import logging
import sys
from grpc import _compression
from grpc._cython import cygrpc as _cygrpc
from grpc._runtime_protos import protos
from grpc._runtime_protos import protos_and_services
from grpc._runtime_protos import services
def unary_unary_rpc_method_handler(behavior, request_deserializer=None, response_serializer=None):
    """Creates an RpcMethodHandler for a unary-unary RPC method.

    Args:
      behavior: The implementation of an RPC that accepts one request
        and returns one response.
      request_deserializer: An optional :term:`deserializer` for request deserialization.
      response_serializer: An optional :term:`serializer` for response serialization.

    Returns:
      An RpcMethodHandler object that is typically used by grpc.Server.
    """
    from grpc import _utilities
    return _utilities.RpcMethodHandler(False, False, request_deserializer, response_serializer, behavior, None, None, None)