import time
from collections import defaultdict
from hashlib import md5
from django.conf import settings
from django.core.cache import caches
from django.http import HttpResponse, HttpResponseNotModified
from django.utils.http import http_date, parse_etags, parse_http_date_safe, quote_etag
from django.utils.log import log_response
from django.utils.regex_helper import _lazy_re_compile
from django.utils.timezone import get_current_timezone_name
from django.utils.translation import get_language
def patch_cache_control(response, **kwargs):
    """
    Patch the Cache-Control header by adding all keyword arguments to it.
    The transformation is as follows:

    * All keyword parameter names are turned to lowercase, and underscores
      are converted to hyphens.
    * If the value of a parameter is True (exactly True, not just a
      true value), only the parameter name is added to the header.
    * All other parameters are added with their value, after applying
      str() to it.
    """

    def dictitem(s):
        t = s.split('=', 1)
        if len(t) > 1:
            return (t[0].lower(), t[1])
        else:
            return (t[0].lower(), True)

    def dictvalue(*t):
        if t[1] is True:
            return t[0]
        else:
            return '%s=%s' % (t[0], t[1])
    cc = defaultdict(set)
    if response.get('Cache-Control'):
        for field in cc_delim_re.split(response.headers['Cache-Control']):
            directive, value = dictitem(field)
            if directive == 'no-cache':
                cc[directive].add(value)
            else:
                cc[directive] = value
    if 'max-age' in cc and 'max_age' in kwargs:
        kwargs['max_age'] = min(int(cc['max-age']), kwargs['max_age'])
    if 'private' in cc and 'public' in kwargs:
        del cc['private']
    elif 'public' in cc and 'private' in kwargs:
        del cc['public']
    for k, v in kwargs.items():
        directive = k.replace('_', '-')
        if directive == 'no-cache':
            cc[directive].add(v)
        else:
            cc[directive] = v
    directives = []
    for directive, values in cc.items():
        if isinstance(values, set):
            if True in values:
                values = {True}
            directives.extend([dictvalue(directive, value) for value in values])
        else:
            directives.append(dictvalue(directive, values))
    cc = ', '.join(directives)
    response.headers['Cache-Control'] = cc