from collections import namedtuple
import datetime
import sys
import struct
@staticmethod
def from_unix_nano(unix_ns):
    """Create a Timestamp from posix timestamp in nanoseconds.

        :param int unix_ns: Posix timestamp in nanoseconds.
        :rtype: Timestamp
        """
    return Timestamp(*divmod(unix_ns, 10 ** 9))