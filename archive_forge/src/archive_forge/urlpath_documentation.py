from typing import cast
from urllib.parse import quote as urlquote, unquote as urlunquote, urlunsplit
from hyperlink import URL as _URL

        The L{repr} of a L{URLPath} is an eval-able expression which will
        construct a similar L{URLPath}.
        