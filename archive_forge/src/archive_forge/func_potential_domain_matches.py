import re
import time
from http.cookiejar import CookieJar as _CookieJar
from http.cookiejar import DefaultCookiePolicy
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.python import to_unicode
def potential_domain_matches(domain):
    """Potential domain matches for a cookie

    >>> potential_domain_matches('www.example.com')
    ['www.example.com', 'example.com', '.www.example.com', '.example.com']

    """
    matches = [domain]
    try:
        start = domain.index('.') + 1
        end = domain.rindex('.')
        while start < end:
            matches.append(domain[start:])
            start = domain.index('.', start) + 1
    except ValueError:
        pass
    return matches + ['.' + d for d in matches]