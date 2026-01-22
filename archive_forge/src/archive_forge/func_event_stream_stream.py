import abc
import collections
import enum
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.common import style  # pylint: disable=unused-import
from grpc.framework.foundation import future  # pylint: disable=unused-import
from grpc.framework.foundation import stream  # pylint: disable=unused-import
@abc.abstractmethod
def event_stream_stream(self, group, method, receiver, abortion_callback, timeout, metadata=None, protocol_options=None):
    """Event-driven invocation of a unary-request-stream-response method.

        Args:
          group: The group identifier of the RPC.
          method: The method identifier of the RPC.
          receiver: A ResponseReceiver to be passed the response data of the RPC.
          abortion_callback: A callback to be called and passed an Abortion value
            in the event of RPC abortion.
          timeout: A duration of time in seconds to allow for the RPC.
          metadata: A metadata value to be passed to the service-side of the RPC.
          protocol_options: A value specified by the provider of a Face interface
            implementation affording custom state and behavior.

        Returns:
          A pair of a Call object for the RPC and a stream.Consumer to which the
            request values of the RPC should be passed.
        """
    raise NotImplementedError()