from __future__ import print_function
from reportlab.graphics.barcode.common import Barcode
from reportlab.lib.utils import asNative
def characters(self):
    """ convert own codewords to characters
        >>> print(' '.join(hex(c)[2:] for c in USPS_4State('01234567094989284321','01234567891').characters))
        dcb 85c 8e4 b06 6dd 1740 17c6 1200 123f 1b2b
        """
    if not self._characters:
        codewords = self.codewords
        fcs = self._fcs
        C = []
        aC = C.append
        table1 = self.table1
        table2 = self.table2
        for i in range(10):
            cw = codewords[i]
            if cw <= 1286:
                c = table1[cw]
            else:
                c = table2[cw - 1287]
            if fcs >> i & 1:
                c = ~c & 8191
            aC(c)
        self._characters = tuple(C)
    return self._characters