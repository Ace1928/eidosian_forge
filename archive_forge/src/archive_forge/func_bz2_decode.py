import codecs
import bz2 # this codec needs the optional bz2 module !
def bz2_decode(input, errors='strict'):
    assert errors == 'strict'
    return (bz2.decompress(input), len(input))