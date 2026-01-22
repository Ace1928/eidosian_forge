import re
import urllib.parse
from .html import _BaseHTMLProcessor
def make_safe_absolute_uri(base, rel=None):
    if not ACCEPTABLE_URI_SCHEMES:
        return _urljoin(base, rel or '')
    if not base:
        return rel or ''
    if not rel:
        try:
            scheme = urllib.parse.urlparse(base)[0]
        except ValueError:
            return ''
        if not scheme or scheme in ACCEPTABLE_URI_SCHEMES:
            return base
        return ''
    uri = _urljoin(base, rel)
    if uri.strip().split(':', 1)[0] not in ACCEPTABLE_URI_SCHEMES:
        return ''
    return uri