from your HTTP server.
import pprint
import re
import socket
import sys
import time
import traceback
import os
import json
import unittest  # pylint: disable=deprecated-module,preferred-module
import warnings
import functools
import http.client
import urllib.parse
from more_itertools.more import always_iterable
import jaraco.functools
def status_matches(self, expected):
    """Check whether actual status matches expected."""
    actual = self.status_code if isinstance(expected, int) else self.status
    return expected == actual