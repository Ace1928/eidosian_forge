import errno
import select
import sys
from functools import partial
def select_wait_for_socket(sock, read=False, write=False, timeout=None):
    if not read and (not write):
        raise RuntimeError('must specify at least one of read=True, write=True')
    rcheck = []
    wcheck = []
    if read:
        rcheck.append(sock)
    if write:
        wcheck.append(sock)
    fn = partial(select.select, rcheck, wcheck, wcheck)
    rready, wready, xready = _retry_on_intr(fn, timeout)
    return bool(rready or wready or xready)