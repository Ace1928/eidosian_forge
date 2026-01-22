from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def getRawString(self):
    r = self.buf[self.idx:self.limit]
    self.idx = self.limit
    return r.tostring()