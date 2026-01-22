from __future__ import absolute_import
import struct
import cramjam
def isValidCompressed(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    ok = True
    try:
        decompress(data)
    except UncompressError as err:
        ok = False
    return ok