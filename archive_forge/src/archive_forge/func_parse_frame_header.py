import struct
import binascii
from .exceptions import (
from .flags import Flag, Flags
@staticmethod
def parse_frame_header(header, strict=False):
    """
        Takes a 9-byte frame header and returns a tuple of the appropriate
        Frame object and the length that needs to be read from the socket.

        This populates the flags field, and determines how long the body is.

        :param strict: Whether to raise an exception when encountering a frame
            not defined by spec and implemented by hyperframe.

        :raises hyperframe.exceptions.UnknownFrameError: If a frame of unknown
            type is received.

        .. versionchanged:: 5.0.0
            Added :param:`strict` to accommodate :class:`ExtensionFrame`
        """
    try:
        fields = _STRUCT_HBBBL.unpack(header)
    except struct.error:
        raise InvalidFrameError('Invalid frame header')
    length = (fields[0] << 8) + fields[1]
    type = fields[2]
    flags = fields[3]
    stream_id = fields[4] & 2147483647
    try:
        frame = FRAMES[type](stream_id)
    except KeyError:
        if strict:
            raise UnknownFrameError(type, length)
        frame = ExtensionFrame(type=type, stream_id=stream_id)
    frame.parse_flags(flags)
    return (frame, length)