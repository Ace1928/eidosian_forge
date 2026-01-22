import collections
from grpc.framework.interfaces.base import base
def completion(terminal_metadata, code, message):
    """Creates a base.Completion aggregating the given operation values.

    Args:
      terminal_metadata: A terminal metadata value for an operaton.
      code: A code value for an operation.
      message: A message value for an operation.

    Returns:
      A base.Completion aggregating the given operation values.
    """
    return _Completion(terminal_metadata, code, message)