import zlib
from gitdb.util import byte_ord
import mmap
from itertools import islice
from functools import reduce
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_text
from gitdb.typ import (
from io import StringIO
def connect_with_next_base(self, bdcl):
    """Connect this chain with the next level of our base delta chunklist.
        The goal in this game is to mark as many of our chunks rigid, hence they
        cannot be changed by any of the upcoming bases anymore. Once all our
        chunks are marked like that, we can stop all processing
        :param bdcl: data chunk list being one of our bases. They must be fed in
            consecutively and in order, towards the earliest ancestor delta
        :return: True if processing was done. Use it to abort processing of
            remaining streams if False is returned"""
    nfc = 0
    dci = 0
    slen = len(self)
    ccl = list()
    while dci < slen:
        dc = self[dci]
        dci += 1
        if dc.data is not None:
            nfc += 1
            continue
        del ccl[:]
        delta_list_slice(bdcl, dc.so, dc.ts, ccl)
        ofs = dc.to - dc.so
        for cdc in ccl:
            cdc.to += ofs
        if len(ccl) == 1:
            self[dci - 1] = ccl[0]
        else:
            post_dci = self[dci:]
            del self[dci - 1:]
            self.extend(ccl)
            self.extend(post_dci)
            slen = len(self)
            dci += len(ccl) - 1
    if nfc == slen:
        return False
    return True