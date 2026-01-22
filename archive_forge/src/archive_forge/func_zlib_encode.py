import codecs
import zlib # this codec needs the optional zlib module !
def zlib_encode(input, errors='strict'):
    assert errors == 'strict'
    return (zlib.compress(input), len(input))