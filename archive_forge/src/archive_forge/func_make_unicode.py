import re
import sys
import six
from six.moves.urllib import parse as urlparse
from routes.util import _url_quote as url_quote, _str_encode, as_unicode
def make_unicode(self, s):
    """Transform the given argument into a unicode string."""
    if isinstance(s, six.text_type):
        return s
    elif isinstance(s, bytes):
        return s.decode(self.encoding)
    elif callable(s):
        return s
    else:
        return six.text_type(s)