import collections
from grpc.framework.interfaces.base import base
def full_subscription(operator, protocol_receiver):
    """Creates a "full" base.Subscription for the given base.Operator.

    Args:
      operator: A base.Operator to be used in an operation.
      protocol_receiver: A base.ProtocolReceiver to be used in an operation.

    Returns:
      A base.Subscription of kind base.Subscription.Kind.FULL wrapping the given
        base.Operator and base.ProtocolReceiver.
    """
    return _Subscription(base.Subscription.Kind.FULL, None, None, operator, protocol_receiver)