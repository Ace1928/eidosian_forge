from suds.properties import Unskin
from suds.transport import *
import base64
from http.cookiejar import CookieJar
import http.client
import socket
import sys
import urllib.request, urllib.error, urllib.parse
import gzip
import zlib
from logging import getLogger
def getcookies(self, fp, u2request):
    """
        Add cookies in the request to the cookiejar.

        @param u2request: A urllib2 request.
        @rtype: u2request: urllib2.Request.

        """
    self.cookiejar.extract_cookies(fp, u2request)