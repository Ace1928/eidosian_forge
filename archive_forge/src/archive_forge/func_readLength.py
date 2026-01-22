from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def readLength(text):
    """Read a dimension measurement: accept "3in", "5cm",
    "72 pt" and so on."""
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        text = text.lower()
        numberText, units = (text[:-2], text[-2:])
        numberText = numberText.strip()
        try:
            number = float(numberText)
        except ValueError:
            raise ValueError("invalid length attribute '%s'" % text)
        try:
            multiplier = {'in': 72, 'cm': 28.3464566929, 'mm': 2.83464566929, 'pt': 1}[units]
        except KeyError:
            raise ValueError("invalid length attribute '%s'" % text)
        return number * multiplier