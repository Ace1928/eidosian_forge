import base64
import calendar
from ipaddress import AddressValueError
from ipaddress import IPv4Address
from ipaddress import IPv6Address
import re
import struct
import time
from urllib.parse import urlparse
from saml2 import time_util
def validate_on_or_after(not_on_or_after, slack):
    if not_on_or_after:
        now = time_util.utc_now()
        nooa = calendar.timegm(time_util.str_to_time(not_on_or_after))
        if now > nooa + slack:
            now_str = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(now))
            raise ResponseLifetimeExceed("Can't use response, too old (now=%s + slack=%d > not_on_or_after=%s" % (now_str, slack, not_on_or_after))
        return nooa
    else:
        return False