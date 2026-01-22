import warnings
import sys
from urllib import parse as urlparse
from paste.recursive import ForwardRequestException, RecursiveMiddleware, RecursionLoop
from paste.util import converters
from paste.response import replace_header
def change_response(status, headers, exc_info=None):
    new_url = None
    parts = status.split(' ')
    try:
        code = int(parts[0])
    except (ValueError, TypeError):
        raise Exception('_StatusBasedRedirect middleware received an invalid status code %s' % repr(parts[0]))
    message = ' '.join(parts[1:])
    new_url = self.mapper(code, message, environ, self.global_conf, self.kw)
    if not (new_url == None or isinstance(new_url, str)):
        raise TypeError('Expected the url to internally redirect to in the _StatusBasedRedirect error_mapperto be a string or None, not %s' % repr(new_url))
    if new_url:
        url.append(new_url)
    code_message.append([code, message])
    return start_response(status, headers, exc_info)