import copy
import datetime
import sys
import inspect
import warnings
import hashlib
from http.client import HTTPMessage
import logging
import shlex
import re
import os
from collections import OrderedDict
from collections.abc import MutableMapping
from math import floor
from botocore.vendored import six
from botocore.exceptions import MD5UnavailableError
from dateutil.tz import tzlocal
from urllib3 import exceptions
from urllib.parse import (
from http.client import HTTPResponse
from io import IOBase as _IOBase
from base64 import encodebytes
from email.utils import formatdate
from itertools import zip_longest
import json
def ensure_unicode(s, encoding=None, errors=None):
    return s