import logging
import httplib2
import six
from six.moves import http_client
from oauth2client import _helpers
def clean_headers(headers):
    """Forces header keys and values to be strings, i.e not unicode.

    The httplib module just concats the header keys and values in a way that
    may make the message header a unicode string, which, if it then tries to
    contatenate to a binary request body may result in a unicode decode error.

    Args:
        headers: dict, A dictionary of headers.

    Returns:
        The same dictionary but with all the keys converted to strings.
    """
    clean = {}
    try:
        for k, v in six.iteritems(headers):
            if not isinstance(k, six.binary_type):
                k = str(k)
            if not isinstance(v, six.binary_type):
                v = str(v)
            clean[_helpers._to_bytes(k)] = _helpers._to_bytes(v)
    except UnicodeEncodeError:
        from oauth2client.client import NonAsciiHeaderError
        raise NonAsciiHeaderError(k, ': ', v)
    return clean