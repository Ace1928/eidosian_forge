from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def put16(self, v):
    if v < 0 or v >= 1 << 16:
        raise ProtocolBufferEncodeError('u16 too big')
    self.buf.append(v >> 0 & 255)
    self.buf.append(v >> 8 & 255)
    return