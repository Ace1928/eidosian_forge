import datetime
import errno
import os
import os.path
import time
def set_timestamp(pb, seconds_since_epoch):
    """Sets a `Timestamp` proto message to a floating point UNIX time.

    This is like `pb.FromNanoseconds(int(seconds_since_epoch * 1e9))` but
    without introducing floating-point error.

    Args:
      pb: A `google.protobuf.Timestamp` message to mutate.
      seconds_since_epoch: A `float`, as returned by `time.time`.
    """
    pb.seconds = int(seconds_since_epoch)
    pb.nanos = int(round(seconds_since_epoch % 1 * 10 ** 9))