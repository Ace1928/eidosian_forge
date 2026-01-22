from libcloud.utils.py3 import parse_qs, urlparse
from libcloud.common.base import Connection
def get_response_object(url, method='GET', headers=None, retry_failed=None):
    """
    Utility function which uses libcloud's connection class to issue an HTTP
    request.

    :param url: URL to send the request to.
    :type url: ``str``

    :param method: HTTP method.
    :type method: ``str``

    :param headers: Optional request headers.
    :type headers: ``dict``

    :param retry_failed: True to retry failed requests.

    :return: Response object.
    :rtype: :class:`Response`.
    """
    parsed_url = urlparse.urlparse(url)
    parsed_qs = parse_qs(parsed_url.query)
    secure = parsed_url.scheme == 'https'
    headers = headers or {}
    method = method.upper()
    con = Connection(secure=secure, host=parsed_url.netloc)
    response = con.request(action=parsed_url.path, params=parsed_qs, headers=headers, method=method, retry_failed=retry_failed)
    return response