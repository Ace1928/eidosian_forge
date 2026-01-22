import enum
import functools
import random
import threading
import time
import grpc
from tensorboard import version
from tensorboard.util import tb_logging
def version_metadata():
    """Creates gRPC invocation metadata encoding the TensorBoard version.

    Usage: `stub.MyRpc(request, metadata=version_metadata())`.

    Returns:
      A tuple of key-value pairs (themselves 2-tuples) to be passed as the
      `metadata` kwarg to gRPC stub API methods.
    """
    return ((_VERSION_METADATA_KEY, version.VERSION),)