import codecs
import datetime
import locale
from decimal import Decimal
from types import NoneType
from urllib.parse import quote
from django.utils.functional import Promise
def punycode(domain):
    """Return the Punycode of the given domain if it's non-ASCII."""
    return domain.encode('idna').decode('ascii')