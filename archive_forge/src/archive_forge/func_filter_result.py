import os
import gzip
import random
import pathlib
import datetime
import contextlib
from http.cookiejar import LWPCookieJar
from urllib.parse import urlparse, parse_qs
from typing import List, Optional, TYPE_CHECKING
def filter_result(link: str) -> Optional[str]:
    """
    Filter links found in the Google result pages HTML code.
    """
    with contextlib.suppress(Exception):
        if link.startswith('/url?'):
            o = urlparse(link, 'http')
            link = parse_qs(o.query)['q'][0]
        o = urlparse(link, 'http')
        if o.netloc and 'google' not in o.netloc:
            return link
    return None