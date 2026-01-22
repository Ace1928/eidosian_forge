import ipaddress
import math
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from django.core.exceptions import ValidationError
from django.utils.deconstruct import deconstructible
from django.utils.encoding import punycode
from django.utils.ipv6 import is_valid_ipv6_address
from django.utils.regex_helper import _lazy_re_compile
from django.utils.translation import gettext_lazy as _
from django.utils.translation import ngettext_lazy
def int_list_validator(sep=',', message=None, code='invalid', allow_negative=False):
    regexp = _lazy_re_compile('^%(neg)s\\d+(?:%(sep)s%(neg)s\\d+)*\\Z' % {'neg': '(-)?' if allow_negative else '', 'sep': re.escape(sep)})
    return RegexValidator(regexp, message=message, code=code)