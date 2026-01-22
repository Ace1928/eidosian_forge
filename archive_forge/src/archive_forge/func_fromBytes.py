from typing import cast
from urllib.parse import quote as urlquote, unquote as urlunquote, urlunsplit
from hyperlink import URL as _URL
@classmethod
def fromBytes(klass, url):
    """
        Make a L{URLPath} from a L{bytes}.

        @param url: A L{bytes} representation of a URL.
        @type url: L{bytes}

        @return: a new L{URLPath} derived from the given L{bytes}.
        @rtype: L{URLPath}

        @since: 15.4
        """
    if not isinstance(url, bytes):
        raise ValueError("'url' must be bytes")
    quoted = urlquote(url, safe=_allascii)
    return klass.fromString(quoted)