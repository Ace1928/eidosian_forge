from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def lengthVarInt32(self, n):
    return self.lengthVarInt64(n)