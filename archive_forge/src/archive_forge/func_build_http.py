from __future__ import absolute_import
import six
from six.moves import http_client
from six.moves import range
from six import BytesIO, StringIO
from six.moves.urllib.parse import urlparse, urlunparse, quote, unquote
import copy
import httplib2
import json
import logging
import mimetypes
import os
import random
import socket
import time
import uuid
from email.generator import Generator
from email.mime.multipart import MIMEMultipart
from email.mime.nonmultipart import MIMENonMultipart
from email.parser import FeedParser
from googleapiclient import _helpers as util
from googleapiclient import _auth
from googleapiclient.errors import BatchError
from googleapiclient.errors import HttpError
from googleapiclient.errors import InvalidChunkSizeError
from googleapiclient.errors import ResumableUploadError
from googleapiclient.errors import UnexpectedBodyError
from googleapiclient.errors import UnexpectedMethodError
from googleapiclient.model import JsonModel
def build_http():
    """Builds httplib2.Http object

  Returns:
  A httplib2.Http object, which is used to make http requests, and which has timeout set by default.
  To override default timeout call

    socket.setdefaulttimeout(timeout_in_sec)

  before interacting with this method.
  """
    if socket.getdefaulttimeout() is not None:
        http_timeout = socket.getdefaulttimeout()
    else:
        http_timeout = DEFAULT_HTTP_TIMEOUT_SEC
    http = httplib2.Http(timeout=http_timeout)
    try:
        http.redirect_codes = http.redirect_codes - {308}
    except AttributeError:
        pass
    return http