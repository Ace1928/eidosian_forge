from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
@staticmethod
def unpack_int8(data):
    """Parse a signed 8-bit integer from the bytes.

        :type data: bytes
        :param data: The bytes to parse from.

        :rtype: (int, int)
        :returns: A tuple containing the (parsed integer value, bytes consumed)
        """
    value = unpack(DecodeUtils.INT8_BYTE_FORMAT, data[:1])[0]
    return (value, 1)