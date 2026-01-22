import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
def test_get_profiler_not_inited(self):
    profiler.clean()
    self.assertIsNone(profiler.get())