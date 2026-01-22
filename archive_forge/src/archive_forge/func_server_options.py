import threading  # pylint: disable=unused-import
import grpc
from grpc import _auth
from grpc.beta import _client_adaptations
from grpc.beta import _metadata
from grpc.beta import _server_adaptations
from grpc.beta import interfaces  # pylint: disable=unused-import
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.interfaces.face import face  # pylint: disable=unused-import
def server_options(multi_method_implementation=None, request_deserializers=None, response_serializers=None, thread_pool=None, thread_pool_size=None, default_timeout=None, maximum_timeout=None):
    """Creates a ServerOptions value to be passed at server creation.

    All parameters are optional and should always be passed by keyword.

    Args:
      multi_method_implementation: A face.MultiMethodImplementation to be called
        to service an RPC if the server has no specific method implementation for
        the name of the RPC for which service was requested.
      request_deserializers: A dictionary from service name-method name pair to
        request deserialization behavior.
      response_serializers: A dictionary from service name-method name pair to
        response serialization behavior.
      thread_pool: A thread pool to use in stubs.
      thread_pool_size: The size of thread pool to create for use in stubs;
        ignored if thread_pool has been passed.
      default_timeout: A duration in seconds to allow for RPC service when
        servicing RPCs that did not include a timeout value when invoked.
      maximum_timeout: A duration in seconds to allow for RPC service when
        servicing RPCs no matter what timeout value was passed when the RPC was
        invoked.

    Returns:
      A StubOptions value created from the passed parameters.
    """
    return ServerOptions(multi_method_implementation, request_deserializers, response_serializers, thread_pool, thread_pool_size, default_timeout, maximum_timeout)