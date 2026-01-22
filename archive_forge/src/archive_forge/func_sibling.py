from typing import cast
from urllib.parse import quote as urlquote, unquote as urlunquote, urlunsplit
from hyperlink import URL as _URL
def sibling(self, path, keepQuery=False):
    """
        Get the sibling of the current L{URLPath}.  A sibling is a file which
        is in the same directory as the current file.

        @param path: The path of the sibling.
        @type path: L{bytes}

        @param keepQuery: Whether to keep the query parameters on the returned
            L{URLPath}.
        @type keepQuery: L{bool}

        @return: a new L{URLPath}
        """
    return self._mod(self._url.sibling(path.decode('ascii')), keepQuery)