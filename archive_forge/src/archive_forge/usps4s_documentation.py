from __future__ import print_function
from reportlab.graphics.barcode.common import Barcode
from reportlab.lib.utils import asNative
Get 4 state bar codes for current routing and tracking
        >>> print(USPS_4State('01234567094989284321','01234567891').barcodes)
        AADTFFDFTDADTAADAATFDTDDAAADDTDTTDAFADADDDTFFFDDTTTADFAAADFTDAADA
        