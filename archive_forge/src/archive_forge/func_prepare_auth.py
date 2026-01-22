import datetime
import sys
import encodings.idna
from urllib3.fields import RequestField
from urllib3.filepost import encode_multipart_formdata
from urllib3.util import parse_url
from urllib3.exceptions import (
from io import UnsupportedOperation
from .hooks import default_hooks
from .structures import CaseInsensitiveDict
from .auth import HTTPBasicAuth
from .cookies import cookiejar_from_dict, get_cookie_header, _copy_cookie_jar
from .exceptions import (
from .exceptions import JSONDecodeError as RequestsJSONDecodeError
from .exceptions import SSLError as RequestsSSLError
from ._internal_utils import to_native_string, unicode_is_ascii
from .utils import (
from .compat import (
from .compat import json as complexjson
from .status_codes import codes
def prepare_auth(self, auth, url=''):
    """Prepares the given HTTP auth data."""
    if auth is None:
        url_auth = get_auth_from_url(self.url)
        auth = url_auth if any(url_auth) else None
    if auth:
        if isinstance(auth, tuple) and len(auth) == 2:
            auth = HTTPBasicAuth(*auth)
        r = auth(self)
        self.__dict__.update(r.__dict__)
        self.prepare_content_length(self.body)