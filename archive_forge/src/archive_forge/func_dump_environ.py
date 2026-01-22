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
def dump_environ(environ, start_response):
    """
    Application which simply dumps the current environment
    variables out as a plain text response.
    """
    output = []
    keys = list(environ.keys())
    keys.sort()
    for k in keys:
        v = str(environ[k]).replace('\n', '\n    ')
        output.append('%s: %s\n' % (k, v))
    output.append('\n')
    content_length = environ.get('CONTENT_LENGTH', '')
    if content_length:
        output.append(environ['wsgi.input'].read(int(content_length)))
        output.append('\n')
    output = ''.join(output)
    output = output.encode('utf8')
    headers = [('Content-Type', 'text/plain'), ('Content-Length', str(len(output)))]
    start_response('200 OK', headers)
    return [output]