import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def make_cookies(self, response, request):
    """Return sequence of Cookie objects extracted from response object."""
    headers = response.info()
    rfc2965_hdrs = headers.get_all('Set-Cookie2', [])
    ns_hdrs = headers.get_all('Set-Cookie', [])
    self._policy._now = self._now = int(time.time())
    rfc2965 = self._policy.rfc2965
    netscape = self._policy.netscape
    if not rfc2965_hdrs and (not ns_hdrs) or (not ns_hdrs and (not rfc2965)) or (not rfc2965_hdrs and (not netscape)) or (not netscape and (not rfc2965)):
        return []
    try:
        cookies = self._cookies_from_attrs_set(split_header_words(rfc2965_hdrs), request)
    except Exception:
        _warn_unhandled_exception()
        cookies = []
    if ns_hdrs and netscape:
        try:
            ns_cookies = self._cookies_from_attrs_set(parse_ns_headers(ns_hdrs), request)
        except Exception:
            _warn_unhandled_exception()
            ns_cookies = []
        self._process_rfc2109_cookies(ns_cookies)
        if rfc2965:
            lookup = {}
            for cookie in cookies:
                lookup[cookie.domain, cookie.path, cookie.name] = None

            def no_matching_rfc2965(ns_cookie, lookup=lookup):
                key = (ns_cookie.domain, ns_cookie.path, ns_cookie.name)
                return key not in lookup
            ns_cookies = filter(no_matching_rfc2965, ns_cookies)
        if ns_cookies:
            cookies.extend(ns_cookies)
    return cookies