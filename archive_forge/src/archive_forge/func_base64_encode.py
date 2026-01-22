import codecs
import base64
def base64_encode(input, errors='strict'):
    assert errors == 'strict'
    return (base64.encodebytes(input), len(input))