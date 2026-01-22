import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
def test_get_profiler_and_init(self):
    p = profiler.init('secret', base_id='1', parent_id='2')
    self.assertEqual(profiler.get(), p)
    self.assertEqual(p.get_base_id(), '1')
    self.assertEqual(p.get_id(), '2')