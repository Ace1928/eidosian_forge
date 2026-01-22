from http.cookies import SimpleCookie
import time
import random
import os
import datetime
import threading
import tempfile
from paste import wsgilib
from paste import request
def make_session_middleware(app, global_conf, session_expiration=NoDefault, expiration=NoDefault, cookie_name=NoDefault, session_file_path=NoDefault, chmod=NoDefault):
    """
    Adds a middleware that handles sessions for your applications.
    The session is a peristent dictionary.  To get this dictionary
    in your application, use ``environ['paste.session.factory']()``
    which returns this persistent dictionary.

    Configuration:

      session_expiration:
          The time each session lives, in minutes.  This controls
          the cookie expiration.  Default 12 hours.

      expiration:
          The time each session lives on disk.  Old sessions are
          culled from disk based on this.  Default 48 hours.

      cookie_name:
          The cookie name used to track the session.  Use different
          names to avoid session clashes.

      session_file_path:
          Sessions are put in this location, default /tmp.

      chmod:
          The octal chmod you want to apply to new sessions (e.g., 660
          to make the sessions group readable/writable)

    Each of these also takes from the global configuration.  cookie_name
    and chmod take from session_cookie_name and session_chmod
    """
    if session_expiration is NoDefault:
        session_expiration = global_conf.get('session_expiration', 60 * 12)
    session_expiration = int(session_expiration)
    if expiration is NoDefault:
        expiration = global_conf.get('expiration', 60 * 48)
    expiration = int(expiration)
    if cookie_name is NoDefault:
        cookie_name = global_conf.get('session_cookie_name', '_SID_')
    if session_file_path is NoDefault:
        session_file_path = global_conf.get('session_file_path', '/tmp')
    if chmod is NoDefault:
        chmod = global_conf.get('session_chmod', None)
    return SessionMiddleware(app, session_expiration=session_expiration, expiration=expiration, cookie_name=cookie_name, session_file_path=session_file_path, chmod=chmod)