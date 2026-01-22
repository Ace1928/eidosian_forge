import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def parse_ns_headers(ns_headers):
    """Ad-hoc parser for Netscape protocol cookie-attributes.

    The old Netscape cookie format for Set-Cookie can for instance contain
    an unquoted "," in the expires field, so we have to use this ad-hoc
    parser instead of split_header_words.

    XXX This may not make the best possible effort to parse all the crap
    that Netscape Cookie headers contain.  Ronald Tschalar's HTTPClient
    parser is probably better, so could do worse than following that if
    this ever gives any trouble.

    Currently, this is also used for parsing RFC 2109 cookies.

    """
    known_attrs = ('expires', 'domain', 'path', 'secure', 'version', 'port', 'max-age')
    result = []
    for ns_header in ns_headers:
        pairs = []
        version_set = False
        for ii, param in enumerate(ns_header.split(';')):
            param = param.strip()
            key, sep, val = param.partition('=')
            key = key.strip()
            if not key:
                if ii == 0:
                    break
                else:
                    continue
            val = val.strip() if sep else None
            if ii != 0:
                lc = key.lower()
                if lc in known_attrs:
                    key = lc
                if key == 'version':
                    if val is not None:
                        val = strip_quotes(val)
                    version_set = True
                elif key == 'expires':
                    if val is not None:
                        val = http2time(strip_quotes(val))
            pairs.append((key, val))
        if pairs:
            if not version_set:
                pairs.append(('version', '0'))
            result.append(pairs)
    return result