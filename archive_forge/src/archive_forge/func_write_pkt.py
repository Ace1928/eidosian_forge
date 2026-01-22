import struct
import sys
import time
def write_pkt(self, buf, ts=None):
    ts = time.time() if ts is None else ts
    buf_len = len(buf)
    if buf_len > self.snaplen:
        buf_len = self.snaplen
        buf = buf[:self.snaplen]
    self._write_pkt_hdr(ts, buf_len)
    self._f.write(buf)