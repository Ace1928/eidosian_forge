import time
import functools
from hashlib import md5
from urllib.request import parse_http_list, parse_keqv_list
import cherrypy
from cherrypy._cpcompat import ntob, tonative
def www_authenticate(realm, key, algorithm='MD5', nonce=None, qop=qop_auth, stale=False, accept_charset=DEFAULT_CHARSET[:]):
    """Constructs a WWW-Authenticate header for Digest authentication."""
    if qop not in valid_qops:
        raise ValueError("Unsupported value for qop: '%s'" % qop)
    if algorithm not in valid_algorithms:
        raise ValueError("Unsupported value for algorithm: '%s'" % algorithm)
    HEADER_PATTERN = 'Digest realm="%s", nonce="%s", algorithm="%s", qop="%s"%s%s'
    if nonce is None:
        nonce = synthesize_nonce(realm, key)
    stale_param = ', stale="true"' if stale else ''
    charset_declaration = _get_charset_declaration(accept_charset)
    return HEADER_PATTERN % (realm, nonce, algorithm, qop, stale_param, charset_declaration)