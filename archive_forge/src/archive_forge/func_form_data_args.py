from tornado.httputil import (
from tornado.escape import utf8, native_str
from tornado.log import gen_log
from tornado.testing import ExpectLog
from tornado.test.util import ignore_deprecation
import copy
import datetime
import logging
import pickle
import time
import urllib.parse
import unittest
from typing import Tuple, Dict, List
def form_data_args() -> Tuple[Dict[str, List[bytes]], Dict[str, List[HTTPFile]]]:
    """Return two empty dicts suitable for use with parse_multipart_form_data.

    mypy insists on type annotations for dict literals, so this lets us avoid
    the verbose types throughout this test.
    """
    return ({}, {})