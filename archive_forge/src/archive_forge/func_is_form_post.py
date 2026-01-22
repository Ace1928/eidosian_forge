import re
import logging
from webob import Request
from routes.base import request_config
from routes.util import URLGenerator
def is_form_post(environ):
    """Determine whether the request is a POSTed html form"""
    content_type = environ.get('CONTENT_TYPE', '').lower()
    if ';' in content_type:
        content_type = content_type.split(';', 1)[0]
    return content_type in ('application/x-www-form-urlencoded', 'multipart/form-data')