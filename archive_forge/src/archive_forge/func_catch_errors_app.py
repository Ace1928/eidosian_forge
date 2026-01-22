import io
import sys
import warnings
from traceback import print_exception
from io import StringIO
from urllib.parse import unquote, urlsplit
from paste.request import get_cookies, parse_querystring, parse_formvars
from paste.request import construct_url, path_info_split, path_info_pop
from paste.response import HeaderDict, has_header, header_value, remove_header
from paste.response import error_body_response, error_response, error_response_app
def catch_errors_app(application, environ, start_response, error_callback_app, ok_callback=None, catch=Exception):
    """
    Like ``catch_errors``, except error_callback_app should be a
    callable that will receive *three* arguments -- ``environ``,
    ``start_response``, and ``exc_info``.  It should call
    ``start_response`` (*with* the exc_info argument!) and return an
    iterator.
    """
    try:
        app_iter = application(environ, start_response)
    except catch:
        return error_callback_app(environ, start_response, sys.exc_info())
    if type(app_iter) in (list, tuple):
        if ok_callback is not None:
            ok_callback()
        return app_iter
    else:
        return _wrap_app_iter_app(environ, start_response, app_iter, error_callback_app, ok_callback, catch=catch)