import re
import six
from genshi.core import Attrs, QName, stripentities
from genshi.core import END, START, TEXT, COMMENT
def is_safe_uri(self, uri):
    """Determine whether the given URI is to be considered safe for
        inclusion in the output.
        
        The default implementation checks whether the scheme of the URI is in
        the set of allowed URIs (`safe_schemes`).
        
        >>> sanitizer = HTMLSanitizer()
        >>> sanitizer.is_safe_uri('http://example.org/')
        True
        >>> sanitizer.is_safe_uri('javascript:alert(document.cookie)')
        False
        
        :param uri: the URI to check
        :return: `True` if the URI can be considered safe, `False` otherwise
        :rtype: `bool`
        :since: version 0.4.3
        """
    if '#' in uri:
        uri = uri.split('#', 1)[0]
    if ':' not in uri:
        return True
    chars = [char for char in uri.split(':', 1)[0] if char.isalnum()]
    return ''.join(chars).lower() in self.safe_schemes