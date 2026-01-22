from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def getPrefixedString(self):
    length = self.getVarInt32()
    if self.idx + length > self.limit:
        raise ProtocolBufferDecodeError('truncated')
    r = self.buf[self.idx:self.idx + length]
    self.idx += length
    return r.tostring()