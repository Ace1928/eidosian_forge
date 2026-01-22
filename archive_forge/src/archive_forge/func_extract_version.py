import enum
import functools
import random
import threading
import time
import grpc
from tensorboard import version
from tensorboard.util import tb_logging
def extract_version(metadata):
    """Extracts version from invocation metadata.

    The argument should be the result of a prior call to `metadata` or the
    result of combining such a result with other metadata.

    Returns:
      The TensorBoard version listed in this metadata, or `None` if none
      is listed.
    """
    return dict(metadata).get(_VERSION_METADATA_KEY)