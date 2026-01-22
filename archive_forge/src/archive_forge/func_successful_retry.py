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
def successful_retry(self, request):
    """Handler which will return an error and then success

        It's not currently very flexible as the number of retries is hard-coded.
        """
    test_name = request.headers.get('test-name', None)
    if not test_name:
        return Response('test-name header not set', status='400 Bad Request')
    RETRY_TEST_NAMES[test_name] += 1
    if RETRY_TEST_NAMES[test_name] >= 2:
        return Response('Retry successful!')
    else:
        return Response('need to keep retrying!', status="418 I'm A Teapot")