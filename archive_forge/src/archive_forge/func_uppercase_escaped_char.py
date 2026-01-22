import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def uppercase_escaped_char(match):
    return '%%%s' % match.group(1).upper()