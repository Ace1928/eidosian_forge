import time
import functools
from hashlib import md5
from urllib.request import parse_http_list, parse_keqv_list
import cherrypy
from cherrypy._cpcompat import ntob, tonative
def digest_auth(realm, get_ha1, key, debug=False, accept_charset='utf-8'):
    """A CherryPy tool that hooks at before_handler to perform
    HTTP Digest Access Authentication, as specified in :rfc:`2617`.

    If the request has an 'authorization' header with a 'Digest' scheme,
    this tool authenticates the credentials supplied in that header.
    If the request has no 'authorization' header, or if it does but the
    scheme is not "Digest", or if authentication fails, the tool sends
    a 401 response with a 'WWW-Authenticate' Digest header.

    realm
        A string containing the authentication realm.

    get_ha1
        A callable that looks up a username in a credentials store
        and returns the HA1 string, which is defined in the RFC to be
        MD5(username : realm : password).  The function's signature is:
        ``get_ha1(realm, username)``
        where username is obtained from the request's 'authorization' header.
        If username is not found in the credentials store, get_ha1() returns
        None.

    key
        A secret string known only to the server, used in the synthesis
        of nonces.

    """
    request = cherrypy.serving.request
    auth_header = request.headers.get('authorization')
    respond_401 = functools.partial(_respond_401, realm, key, accept_charset, debug)
    if not HttpDigestAuthorization.matches(auth_header or ''):
        respond_401()
    msg = 'The Authorization header could not be parsed.'
    with cherrypy.HTTPError.handle(ValueError, 400, msg):
        auth = HttpDigestAuthorization(auth_header, request.method, debug=debug, accept_charset=accept_charset)
    if debug:
        TRACE(str(auth))
    if not auth.validate_nonce(realm, key):
        respond_401()
    ha1 = get_ha1(realm, auth.username)
    if ha1 is None:
        respond_401()
    digest = auth.request_digest(ha1, entity_body=request.body)
    if digest != auth.response:
        respond_401()
    if debug:
        TRACE('digest matches auth.response')
    if auth.is_nonce_stale(max_age_seconds=600):
        respond_401(stale=True)
    request.login = auth.username
    if debug:
        TRACE('authentication of %s successful' % auth.username)