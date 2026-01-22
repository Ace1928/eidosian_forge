import abc
import collections
import enum
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.common import style  # pylint: disable=unused-import
from grpc.framework.foundation import future  # pylint: disable=unused-import
from grpc.framework.foundation import stream  # pylint: disable=unused-import
@abc.abstractmethod
def inline_unary_stream(self, group, method, request, timeout, metadata=None, protocol_options=None):
    """Invokes a unary-request-stream-response method.

        Args:
          group: The group identifier of the RPC.
          method: The method identifier of the RPC.
          request: The request value for the RPC.
          timeout: A duration of time in seconds to allow for the RPC.
          metadata: A metadata value to be passed to the service-side of the RPC.
          protocol_options: A value specified by the provider of a Face interface
            implementation affording custom state and behavior.

        Returns:
          An object that is both a Call for the RPC and an iterator of response
            values. Drawing response values from the returned iterator may raise
            AbortionError indicating abortion of the RPC.
        """
    raise NotImplementedError()