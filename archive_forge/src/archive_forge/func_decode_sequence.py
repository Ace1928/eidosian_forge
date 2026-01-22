from paramiko.common import max_byte, zero_byte, byte_ord, byte_chr
import paramiko.util as util
from paramiko.util import b
from paramiko.sftp import int64
@staticmethod
def decode_sequence(data):
    out = []
    ber = BER(data)
    while True:
        x = ber.decode_next()
        if x is None:
            break
        out.append(x)
    return out