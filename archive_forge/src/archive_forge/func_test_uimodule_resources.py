from tornado.concurrent import Future
from tornado import gen
from tornado.escape import (
from tornado.httpclient import HTTPClientError
from tornado.httputil import format_timestamp
from tornado.iostream import IOStream
from tornado import locale
from tornado.locks import Event
from tornado.log import app_log, gen_log
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.template import DictLoader
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, ExpectLog, gen_test
from tornado.test.util import ignore_deprecation
from tornado.util import ObjectDict, unicode_type
from tornado.web import (
import binascii
import contextlib
import copy
import datetime
import email.utils
import gzip
from io import BytesIO
import itertools
import logging
import os
import re
import socket
import typing  # noqa: F401
import unittest
import urllib.parse
def test_uimodule_resources(self):
    response = self.fetch('/uimodule_resources')
    self.assertEqual(response.body, b'<html><head><link href="/base.css" type="text/css" rel="stylesheet"/><link href="/foo.css" type="text/css" rel="stylesheet"/>\n<style type="text/css">\n.entry { margin-bottom: 1em; }\n</style>\n<meta>\n</head><body>\n\n\n<div class="entry">...</div>\n\n\n<div class="entry">...</div>\n\n<script src="/common.js" type="text/javascript"></script>\n<script type="text/javascript">\n//<![CDATA[\njs_embed()\n//]]>\n</script>\n<script src="/analytics.js"/>\n</body></html>')