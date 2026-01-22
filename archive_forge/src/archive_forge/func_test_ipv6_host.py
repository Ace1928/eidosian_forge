import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
def test_ipv6_host(self):
    url = 'rsync://[2001:db8:0:1::2]:873'
    parsed = misc.parse_uri(url)
    self.assertEqual('rsync', parsed.scheme)
    self.assertEqual('2001:db8:0:1::2', parsed.hostname)
    self.assertEqual(873, parsed.port)