from unicodedata import category
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.rl_config import _FUZZ
from reportlab.lib.utils import isUnicode
import re
def is_multi_byte(ch):
    """Is this an Asian character?"""
    return ord(ch) >= 12288