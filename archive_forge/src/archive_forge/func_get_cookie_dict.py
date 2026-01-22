import cgi
from collections.abc import MutableMapping as DictMixin
from urllib import parse as urlparse
from urllib.parse import quote, parse_qsl
from http.cookies import SimpleCookie, CookieError
from paste.util.multidict import MultiDict
def get_cookie_dict(environ):
    """Return a *plain* dictionary of cookies as found in the request.

    Unlike ``get_cookies`` this returns a dictionary, not a
    ``SimpleCookie`` object.  For incoming cookies a dictionary fully
    represents the information.  Like ``get_cookies`` this caches and
    checks the cache.
    """
    header = environ.get('HTTP_COOKIE')
    if not header:
        return {}
    if 'paste.cookies.dict' in environ:
        cookies, check_header = environ['paste.cookies.dict']
        if check_header == header:
            return cookies
    cookies = SimpleCookie()
    try:
        cookies.load(header)
    except CookieError:
        pass
    result = {}
    for name in cookies:
        result[name] = cookies[name].value
    environ['paste.cookies.dict'] = (result, header)
    return result