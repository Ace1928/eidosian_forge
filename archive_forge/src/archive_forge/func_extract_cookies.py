import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def extract_cookies(self, response, request):
    """Extract cookies from response, where allowable given the request."""
    _debug('extract_cookies: %s', response.info())
    self._cookies_lock.acquire()
    try:
        for cookie in self.make_cookies(response, request):
            if self._policy.set_ok(cookie, request):
                _debug(' setting cookie: %s', cookie)
                self.set_cookie(cookie)
    finally:
        self._cookies_lock.release()