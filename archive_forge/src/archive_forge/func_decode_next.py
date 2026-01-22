from paramiko.common import max_byte, zero_byte, byte_ord, byte_chr
import paramiko.util as util
from paramiko.util import b
from paramiko.sftp import int64
def decode_next(self):
    if self.idx >= len(self.content):
        return None
    ident = byte_ord(self.content[self.idx])
    self.idx += 1
    if ident & 31 == 31:
        ident = 0
        while self.idx < len(self.content):
            t = byte_ord(self.content[self.idx])
            self.idx += 1
            ident = ident << 7 | t & 127
            if not t & 128:
                break
    if self.idx >= len(self.content):
        return None
    size = byte_ord(self.content[self.idx])
    self.idx += 1
    if size & 128:
        t = size & 127
        if self.idx + t > len(self.content):
            return None
        size = util.inflate_long(self.content[self.idx:self.idx + t], True)
        self.idx += t
    if self.idx + size > len(self.content):
        return None
    data = self.content[self.idx:self.idx + size]
    self.idx += size
    if ident == 48:
        return self.decode_sequence(data)
    elif ident == 2:
        return util.inflate_long(data)
    else:
        msg = 'Unknown ber encoding type {:d} (robey is lazy)'
        raise BERException(msg.format(ident))