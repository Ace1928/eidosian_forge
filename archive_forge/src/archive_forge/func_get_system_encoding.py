import codecs
import datetime
import locale
from decimal import Decimal
from types import NoneType
from urllib.parse import quote
from django.utils.functional import Promise
def get_system_encoding():
    """
    The encoding for the character type functions. Fallback to 'ascii' if the
    #encoding is unsupported by Python or could not be determined. See tickets
    #10335 and #5846.
    """
    try:
        encoding = locale.getlocale()[1] or 'ascii'
        codecs.lookup(encoding)
    except Exception:
        encoding = 'ascii'
    return encoding