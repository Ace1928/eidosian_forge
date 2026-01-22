import codecs
import sys
from future import utils
def replace_surrogate_encode(mystring):
    """
    Returns a (unicode) string, not the more logical bytes, because the codecs
    register_error functionality expects this.
    """
    decoded = []
    for ch in mystring:
        code = ord(ch)
        if not 55296 <= code <= 56575:
            raise NotASurrogateError
        if 56320 <= code <= 56447:
            decoded.append(_unichr(code - 56320))
        elif code <= 56575:
            decoded.append(_unichr(code - 56320))
        else:
            raise NotASurrogateError
    return str().join(decoded)