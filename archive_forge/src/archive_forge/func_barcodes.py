from __future__ import print_function
from reportlab.graphics.barcode.common import Barcode
from reportlab.lib.utils import asNative
def barcodes(self):
    """Get 4 state bar codes for current routing and tracking
        >>> print(USPS_4State('01234567094987654321','01234567891').barcodes)
        AADTFFDFTDADTAADAATFDTDDAAADDTDTTDAFADADDDTFFFDDTTTADFAAADFTDAADA
        """
    if not self._barcodes:
        C = self.characters
        B = []
        aB = B.append
        bits2bars = self._bits2bars
        for dc, db, ac, ab in self.table4:
            aB(bits2bars[(C[dc] >> db & 1) + 2 * (C[ac] >> ab & 1)])
        self._barcodes = ''.join(B)
    return self._barcodes