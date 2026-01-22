import abc
import collections
import enum
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.common import style  # pylint: disable=unused-import
from grpc.framework.foundation import future  # pylint: disable=unused-import
from grpc.framework.foundation import stream  # pylint: disable=unused-import
@abc.abstractmethod
def blocking_unary_unary(self, group, method, request, timeout, metadata=None, with_call=False, protocol_options=None):
    """Invokes a unary-request-unary-response method.

        This method blocks until either returning the response value of the RPC
        (in the event of RPC completion) or raising an exception (in the event of
        RPC abortion).

        Args:
          group: The group identifier of the RPC.
          method: The method identifier of the RPC.
          request: The request value for the RPC.
          timeout: A duration of time in seconds to allow for the RPC.
          metadata: A metadata value to be passed to the service-side of the RPC.
          with_call: Whether or not to include return a Call for the RPC in addition
            to the response.
          protocol_options: A value specified by the provider of a Face interface
            implementation affording custom state and behavior.

        Returns:
          The response value for the RPC, and a Call for the RPC if with_call was
            set to True at invocation.

        Raises:
          AbortionError: Indicating that the RPC was aborted.
        """
    raise NotImplementedError()