import codecs
import binascii
def hex_encode(input, errors='strict'):
    assert errors == 'strict'
    return (binascii.b2a_hex(input), len(input))