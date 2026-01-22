import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
def test_profiler_get_shorten_id(self):
    uuid_id = '4e3e0ec6-2938-40b1-8504-09eb1d4b0dee'
    prof = profiler._Profiler('secret', base_id='1', parent_id='2')
    result = prof.get_shorten_id(uuid_id)
    expected = '850409eb1d4b0dee'
    self.assertEqual(expected, result)