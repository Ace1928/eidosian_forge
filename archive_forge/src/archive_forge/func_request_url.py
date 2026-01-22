import os.path
import socket
from urllib3.poolmanager import PoolManager, proxy_from_url
from urllib3.response import HTTPResponse
from urllib3.util import parse_url
from urllib3.util import Timeout as TimeoutSauce
from urllib3.util.retry import Retry
from urllib3.exceptions import ClosedPoolError
from urllib3.exceptions import ConnectTimeoutError
from urllib3.exceptions import HTTPError as _HTTPError
from urllib3.exceptions import InvalidHeader as _InvalidHeader
from urllib3.exceptions import MaxRetryError
from urllib3.exceptions import NewConnectionError
from urllib3.exceptions import ProxyError as _ProxyError
from urllib3.exceptions import ProtocolError
from urllib3.exceptions import ReadTimeoutError
from urllib3.exceptions import SSLError as _SSLError
from urllib3.exceptions import ResponseError
from urllib3.exceptions import LocationValueError
from .models import Response
from .compat import urlparse, basestring
from .utils import (DEFAULT_CA_BUNDLE_PATH, extract_zipped_paths,
from .structures import CaseInsensitiveDict
from .cookies import extract_cookies_to_jar
from .exceptions import (ConnectionError, ConnectTimeout, ReadTimeout, SSLError,
from .auth import _basic_auth_str
def request_url(self, request, proxies):
    """Obtain the url to use when making the final request.

        If the message is being sent through a HTTP proxy, the full URL has to
        be used. Otherwise, we should only use the path portion of the URL.

        This should not be called from user code, and is only exposed for use
        when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param request: The :class:`PreparedRequest <PreparedRequest>` being sent.
        :param proxies: A dictionary of schemes or schemes and hosts to proxy URLs.
        :rtype: str
        """
    proxy = select_proxy(request.url, proxies)
    scheme = urlparse(request.url).scheme
    is_proxied_http_request = proxy and scheme != 'https'
    using_socks_proxy = False
    if proxy:
        proxy_scheme = urlparse(proxy).scheme.lower()
        using_socks_proxy = proxy_scheme.startswith('socks')
    url = request.path_url
    if is_proxied_http_request and (not using_socks_proxy):
        url = urldefragauth(request.url)
    return url