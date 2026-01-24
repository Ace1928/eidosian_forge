from __future__ import print_function
from reportlab.graphics.barcode.common import Barcode
from reportlab.lib.utils import asNative
def codewords(self):
    """convert binary value into codewords
        >>> print(USPS_4State('01234567094989284321','01234567891').codewords)
        (673, 787, 607, 1022, 861, 19, 816, 1294, 35, 602)
        """
    if not self._codewords:
        value = self.binary
        A, J = divmod(value, 636)
        A, I = divmod(A, 1365)
        A, H = divmod(A, 1365)
        A, G = divmod(A, 1365)
        A, F = divmod(A, 1365)
        A, E = divmod(A, 1365)
        A, D = divmod(A, 1365)
        A, C = divmod(A, 1365)
        A, B = divmod(A, 1365)
        assert 0 <= A <= 658, 'improper value %s passed to _2codewords A-->%s' % (hex(int(value)), A)
        self._fcs = _crc11(value)
        if self._fcs & 1024:
            A += 659
        J *= 2
        self._codewords = tuple(map(int, (A, B, C, D, E, F, G, H, I, J)))
    return self._codewords