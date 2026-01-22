import base64
import binascii
import hashlib
import hmac
import time
import urllib.parse
import uuid
import warnings
from tornado import httpclient
from tornado import escape
from tornado.httputil import url_concat
from tornado.util import unicode_type
from tornado.web import RequestHandler
from typing import List, Any, Dict, cast, Iterable, Union, Optional
def authenticate_redirect(self, callback_uri: Optional[str]=None, ax_attrs: List[str]=['name', 'email', 'language', 'username']) -> None:
    """Redirects to the authentication URL for this service.

        After authentication, the service will redirect back to the given
        callback URI with additional parameters including ``openid.mode``.

        We request the given attributes for the authenticated user by
        default (name, email, language, and username). If you don't need
        all those attributes for your app, you can request fewer with
        the ax_attrs keyword argument.

        .. versionchanged:: 6.0

            The ``callback`` argument was removed and this method no
            longer returns an awaitable object. It is now an ordinary
            synchronous function.
        """
    handler = cast(RequestHandler, self)
    callback_uri = callback_uri or handler.request.uri
    assert callback_uri is not None
    args = self._openid_args(callback_uri, ax_attrs=ax_attrs)
    endpoint = self._OPENID_ENDPOINT
    handler.redirect(endpoint + '?' + urllib.parse.urlencode(args))