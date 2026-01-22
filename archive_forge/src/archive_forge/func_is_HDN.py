import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def is_HDN(text):
    """Return True if text is a host domain name."""
    if IPV4_RE.search(text):
        return False
    if text == '':
        return False
    if text[0] == '.' or text[-1] == '.':
        return False
    return True