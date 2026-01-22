import threading  # pylint: disable=unused-import
import grpc
from grpc import _auth
from grpc.beta import _client_adaptations
from grpc.beta import _metadata
from grpc.beta import _server_adaptations
from grpc.beta import interfaces  # pylint: disable=unused-import
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.interfaces.face import face  # pylint: disable=unused-import
def dynamic_stub(channel, service, cardinalities, options=None):
    """Creates a face.DynamicStub with which RPCs can be invoked.

    Args:
      channel: A Channel for the returned face.DynamicStub to use.
      service: The package-qualified full name of the service.
      cardinalities: A dictionary from RPC method name to cardinality.Cardinality
        value identifying the cardinality of the RPC method.
      options: An optional StubOptions value further customizing the functionality
        of the returned face.DynamicStub.

    Returns:
      A face.DynamicStub with which RPCs can be invoked.
    """
    effective_options = _EMPTY_STUB_OPTIONS if options is None else options
    return _client_adaptations.dynamic_stub(channel._channel, service, cardinalities, effective_options.host, effective_options.metadata_transformer, effective_options.request_serializers, effective_options.response_deserializers)