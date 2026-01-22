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
def validate_domain_part(self, domain_part):
    if self.domain_regex.match(domain_part):
        return True
    literal_match = self.literal_regex.match(domain_part)
    if literal_match:
        ip_address = literal_match[1]
        try:
            validate_ipv46_address(ip_address)
            return True
        except ValidationError:
            pass
    return False