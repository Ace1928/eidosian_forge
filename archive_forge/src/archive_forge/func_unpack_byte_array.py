from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
@staticmethod
def unpack_byte_array(data, length_byte_size=2):
    """Parse a variable length byte array from the bytes.

        The bytes are expected to be in the following format:
            [ length ][0 ... length bytes]
        where length is an unsigned integer represented in the smallest number
        of bytes to hold the maximum length of the array.

        :type data: bytes
        :param data: The bytes to parse from.

        :type length_byte_size: int
        :param length_byte_size: The byte size of the preceding integer that
        represents the length of the array. Supported values are 1, 2, and 4.

        :rtype: (bytes, int)
        :returns: A tuple containing the (parsed byte array, bytes consumed).
        """
    uint_byte_format = DecodeUtils.UINT_BYTE_FORMAT[length_byte_size]
    length = unpack(uint_byte_format, data[:length_byte_size])[0]
    bytes_end = length + length_byte_size
    array_bytes = data[length_byte_size:bytes_end]
    return (array_bytes, bytes_end)