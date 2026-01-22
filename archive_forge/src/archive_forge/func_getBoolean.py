from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def getBoolean(self):
    b = self.get8()
    if b != 0 and b != 1:
        raise ProtocolBufferDecodeError('corrupted')
    return b