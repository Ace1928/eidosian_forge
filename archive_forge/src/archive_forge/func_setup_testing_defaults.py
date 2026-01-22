import posixpath
def setup_testing_defaults(environ):
    """Update 'environ' with trivial defaults for testing purposes

    This adds various parameters required for WSGI, including HTTP_HOST,
    SERVER_NAME, SERVER_PORT, REQUEST_METHOD, SCRIPT_NAME, PATH_INFO,
    and all of the wsgi.* variables.  It only supplies default values,
    and does not replace any existing settings for these variables.

    This routine is intended to make it easier for unit tests of WSGI
    servers and applications to set up dummy environments.  It should *not*
    be used by actual WSGI servers or applications, since the data is fake!
    """
    environ.setdefault('SERVER_NAME', '127.0.0.1')
    environ.setdefault('SERVER_PROTOCOL', 'HTTP/1.0')
    environ.setdefault('HTTP_HOST', environ['SERVER_NAME'])
    environ.setdefault('REQUEST_METHOD', 'GET')
    if 'SCRIPT_NAME' not in environ and 'PATH_INFO' not in environ:
        environ.setdefault('SCRIPT_NAME', '')
        environ.setdefault('PATH_INFO', '/')
    environ.setdefault('wsgi.version', (1, 0))
    environ.setdefault('wsgi.run_once', 0)
    environ.setdefault('wsgi.multithread', 0)
    environ.setdefault('wsgi.multiprocess', 0)
    from io import StringIO, BytesIO
    environ.setdefault('wsgi.input', BytesIO())
    environ.setdefault('wsgi.errors', StringIO())
    environ.setdefault('wsgi.url_scheme', guess_scheme(environ))
    if environ['wsgi.url_scheme'] == 'http':
        environ.setdefault('SERVER_PORT', '80')
    elif environ['wsgi.url_scheme'] == 'https':
        environ.setdefault('SERVER_PORT', '443')