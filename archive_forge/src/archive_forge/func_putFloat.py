from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def putFloat(self, v):
    a = array.array('B')
    a.fromstring(struct.pack('<f', v))
    self.buf.extend(a)
    return