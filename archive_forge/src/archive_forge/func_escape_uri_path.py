import codecs
import datetime
import locale
from decimal import Decimal
from types import NoneType
from urllib.parse import quote
from django.utils.functional import Promise
def escape_uri_path(path):
    """
    Escape the unsafe characters from the path portion of a Uniform Resource
    Identifier (URI).
    """
    return quote(path, safe="/:@&+$,-_.!~*'()")