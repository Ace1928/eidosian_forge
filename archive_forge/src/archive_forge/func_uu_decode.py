import codecs
import binascii
from io import BytesIO
def uu_decode(input, errors='strict'):
    assert errors == 'strict'
    infile = BytesIO(input)
    outfile = BytesIO()
    readline = infile.readline
    write = outfile.write
    while 1:
        s = readline()
        if not s:
            raise ValueError('Missing "begin" line in input data')
        if s[:5] == b'begin':
            break
    while True:
        s = readline()
        if not s or s == b'end\n':
            break
        try:
            data = binascii.a2b_uu(s)
        except binascii.Error as v:
            nbytes = ((s[0] - 32 & 63) * 4 + 5) // 3
            data = binascii.a2b_uu(s[:nbytes])
        write(data)
    if not s:
        raise ValueError('Truncated input data')
    return (outfile.getvalue(), len(input))