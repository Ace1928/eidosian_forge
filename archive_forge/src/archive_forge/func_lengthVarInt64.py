from __future__ import absolute_import
import array
import six.moves.http_client
import itertools
import re
import struct
import six
def lengthVarInt64(self, n):
    if n < 0:
        return 10
    result = 0
    while 1:
        result += 1
        n >>= 7
        if n == 0:
            break
    return result