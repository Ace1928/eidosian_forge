from hashlib import md5
import array
import re
def decode_torsion(utf8):
    return [ord(x) for x in utf8.decode('utf8')]