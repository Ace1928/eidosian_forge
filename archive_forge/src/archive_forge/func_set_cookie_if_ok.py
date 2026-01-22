import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def set_cookie_if_ok(self, cookie, request):
    """Set a cookie if policy says it's OK to do so."""
    self._cookies_lock.acquire()
    try:
        self._policy._now = self._now = int(time.time())
        if self._policy.set_ok(cookie, request):
            self.set_cookie(cookie)
    finally:
        self._cookies_lock.release()