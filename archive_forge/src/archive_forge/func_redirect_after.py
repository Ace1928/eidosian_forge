from __future__ import print_function
import collections
import contextlib
import gzip
import json
import logging
import sys
import time
import zlib
from datetime import datetime, timedelta
from io import BytesIO
from tornado import httputil
from tornado.web import RequestHandler
from urllib3.packages.six import binary_type, ensure_str
from urllib3.packages.six.moves.http_client import responses
from urllib3.packages.six.moves.urllib.parse import urlsplit
def redirect_after(self, request):
    """Perform a redirect to ``target``"""
    date = request.params.get('date')
    if date:
        retry_after = str(httputil.format_timestamp(datetime.utcfromtimestamp(float(date))))
    else:
        retry_after = '1'
    target = request.params.get('target', '/')
    headers = [('Location', target), ('Retry-After', retry_after)]
    return Response(status='303 See Other', headers=headers)