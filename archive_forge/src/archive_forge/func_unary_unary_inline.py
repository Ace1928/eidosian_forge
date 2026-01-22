import collections
from grpc.framework.common import cardinality
from grpc.framework.common import style
from grpc.framework.foundation import stream  # pylint: disable=unused-import
from grpc.framework.interfaces.face import face
def unary_unary_inline(behavior):
    """Creates an face.MethodImplementation for the given behavior.

    Args:
      behavior: The implementation of a unary-unary RPC method as a callable value
        that takes a request value and an face.ServicerContext object and
        returns a response value.

    Returns:
      An face.MethodImplementation derived from the given behavior.
    """
    return _MethodImplementation(cardinality.Cardinality.UNARY_UNARY, style.Service.INLINE, behavior, None, None, None, None, None, None, None)