import re
import sys
import warnings

    When applied between a WSGI server and a WSGI application, this
    middleware will check for WSGI compliance on a number of levels.
    This middleware does not modify the request or response in any
    way, but will raise an AssertionError if anything seems off
    (except for a failure to close the application iterator, which
    will be printed to stderr -- there's no way to raise an exception
    at that point).
    