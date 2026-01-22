import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
def test_profiler_hmac(self):
    hmac = 'secret'
    prof = profiler._Profiler(hmac, base_id='1', parent_id='2')
    self.assertEqual(hmac, prof.hmac_key)