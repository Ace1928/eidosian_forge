import paste.util.threadinglocal as threadinglocal
def get_request_id(self, environ):
    """Return a unique identifier for the current request"""
    from paste.evalexception.middleware import get_debug_count
    return get_debug_count(environ)