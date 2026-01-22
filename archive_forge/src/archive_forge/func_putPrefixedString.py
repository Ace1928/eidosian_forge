from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def putPrefixedString(self, v):
    v = str(v)
    self.putVarInt32(len(v))
    self.buf.fromstring(v)
    return