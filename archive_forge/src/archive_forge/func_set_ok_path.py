import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def set_ok_path(self, cookie, request):
    if cookie.path_specified:
        req_path = request_path(request)
        if (cookie.version > 0 or (cookie.version == 0 and self.strict_ns_set_path)) and (not self.path_return_ok(cookie.path, request)):
            _debug('   path attribute %s is not a prefix of request path %s', cookie.path, req_path)
            return False
    return True