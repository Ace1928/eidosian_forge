import codecs
import binascii
from io import BytesIO
def uu_encode(input, errors='strict', filename='<data>', mode=438):
    assert errors == 'strict'
    infile = BytesIO(input)
    outfile = BytesIO()
    read = infile.read
    write = outfile.write
    filename = filename.replace('\n', '\\n')
    filename = filename.replace('\r', '\\r')
    write(('begin %o %s\n' % (mode & 511, filename)).encode('ascii'))
    chunk = read(45)
    while chunk:
        write(binascii.b2a_uu(chunk))
        chunk = read(45)
    write(b' \nend\n')
    return (outfile.getvalue(), len(input))