from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
@property
def payload_end(self):
    """Calculates the byte offset for the end of the message payload.

        The extra minus 4 bytes is for the message CRC.

        :rtype: int
        :returns: The byte offset from the beginning of the event stream
        message to the end of the payload.
        """
    return self.total_length - 4