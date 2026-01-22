import hmac, base64, random, time, warnings
from functools import reduce
from paste.request import get_cookies
def make_auth_cookie(app, global_conf, cookie_name='PASTE_AUTH_COOKIE', scanlist=('REMOTE_USER', 'REMOTE_SESSION'), secret=None, timeout=30, maxlen=4096):
    """
    This middleware uses cookies to stash-away a previously
    authenticated user (and perhaps other variables) so that
    re-authentication is not needed.  This does not implement
    sessions; and therefore N servers can be syncronized to accept the
    same saved authentication if they all use the same cookie_name and
    secret.

    By default, this handler scans the `environ` for the REMOTE_USER
    and REMOTE_SESSION key; if found, it is stored. It can be
    configured to scan other `environ` keys as well -- but be careful
    not to exceed 2-3k (so that the encoded and signed cookie does not
    exceed 4k). You can ask it to handle other environment variables
    by doing:

       ``environ['paste.auth.cookie'].append('your.environ.variable')``

    Configuration:

        ``cookie_name``

            The name of the cookie used to store this content, by
            default it is ``PASTE_AUTH_COOKIE``.

        ``scanlist``

            This is the initial set of ``environ`` keys to
            save/restore to the signed cookie.  By default is consists
            only of ``REMOTE_USER`` and ``REMOTE_SESSION``; any
            space-separated list of environment keys will work.
            However, be careful, as the total saved size is limited to
            around 3k.

        ``secret``

            The secret that will be used to sign the cookies.  If you
            don't provide one (and none is set globally) then a random
            secret will be created.  Each time the server is restarted
            a new secret will then be created and all cookies will
            become invalid!  This can be any string value.

        ``timeout``

            The time to keep the cookie, expressed in minutes.  This
            is handled server-side, so a new cookie with a new timeout
            is added to every response.

        ``maxlen``

            The maximum length of the cookie that is sent (default 4k,
            which is a typical browser maximum)

    """
    if isinstance(scanlist, str):
        scanlist = scanlist.split()
    if secret is None and global_conf.get('secret'):
        secret = global_conf['secret']
    try:
        timeout = int(timeout)
    except ValueError:
        raise ValueError('Bad value for timeout (must be int): %r' % timeout)
    try:
        maxlen = int(maxlen)
    except ValueError:
        raise ValueError('Bad value for maxlen (must be int): %r' % maxlen)
    return AuthCookieHandler(app, cookie_name=cookie_name, scanlist=scanlist, secret=secret, timeout=timeout, maxlen=maxlen)