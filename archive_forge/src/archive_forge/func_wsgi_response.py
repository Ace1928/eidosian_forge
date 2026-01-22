import json
from string import Template
import re
import sys
from webob.acceptparse import create_accept_header
from webob.compat import (
from webob.request import Request
from webob.response import Response
from webob.util import html_escape
@property
def wsgi_response(self):
    return self