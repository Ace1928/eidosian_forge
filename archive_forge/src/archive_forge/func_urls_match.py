import re
from humanfriendly.compat import HTMLParser, StringIO, name2codepoint, unichr
from humanfriendly.text import compact_empty_lines
from humanfriendly.terminal import ANSI_COLOR_CODES, ANSI_RESET, ansi_style
def urls_match(self, a, b):
    """
        Compare two URLs for equality using :func:`normalize_url()`.

        :param a: A string containing a URL.
        :param b: A string containing a URL.
        :returns: :data:`True` if the URLs are the same, :data:`False` otherwise.

        This method is used by :func:`handle_endtag()` to omit the URL of a
        hyperlink (``<a href="...">``) when the link text is that same URL.
        """
    return self.normalize_url(a) == self.normalize_url(b)