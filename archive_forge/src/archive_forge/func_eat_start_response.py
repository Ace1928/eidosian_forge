import warnings
import sys
from urllib import parse as urlparse
from paste.recursive import ForwardRequestException, RecursiveMiddleware, RecursionLoop
from paste.util import converters
from paste.response import replace_header
def eat_start_response(status, headers, exc_info=None):
    """
                    We don't want start_response to do anything since it
                    has already been called
                    """
    if status[:3] != '200':
        raise InvalidForward("The URL %s to internally forward to in order to create an error document did not return a '200' status code." % url_)