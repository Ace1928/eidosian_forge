import os
import gzip
import random
import pathlib
import datetime
import contextlib
from http.cookiejar import LWPCookieJar
from urllib.parse import urlparse, parse_qs
from typing import List, Optional, TYPE_CHECKING
def load_cookie_jar() -> LWPCookieJar:
    """
    Load the cookie jar
    """
    global _cookie_jar
    if _cookie_jar is None:
        _cookie_jar = LWPCookieJar(os.path.join(home_folder or lib_path.as_posix(), '.google-cookie'))
        with contextlib.suppress(Exception):
            _cookie_jar.load()
    return _cookie_jar