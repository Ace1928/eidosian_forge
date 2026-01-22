import codecs
import datetime
import doctest
import io
from itertools import chain
from itertools import combinations
import os
import platform
from queue import Queue
import re
import shutil
import sys
import tempfile
import threading
from unittest import TestSuite
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.content_type import ContentType, UTF8_TEXT
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.tests.helpers import (
from testtools.testresult.doubles import (
from testtools.testresult.real import (
def test_only_one_test_at_a_time(self):
    [result1, result2], events = self.make_results(2)
    test1, test2 = (self, make_test())
    start_time1 = datetime.datetime.utcfromtimestamp(1.489)
    end_time1 = datetime.datetime.utcfromtimestamp(2.476)
    start_time2 = datetime.datetime.utcfromtimestamp(3.489)
    end_time2 = datetime.datetime.utcfromtimestamp(4.489)
    result1.time(start_time1)
    result2.time(start_time2)
    result1.startTest(test1)
    result2.startTest(test2)
    result1.time(end_time1)
    result2.time(end_time2)
    result2.addSuccess(test2)
    result1.addSuccess(test1)
    self.assertEqual([('time', start_time2), ('startTest', test2), ('time', end_time2), ('addSuccess', test2), ('stopTest', test2), ('time', start_time1), ('startTest', test1), ('time', end_time1), ('addSuccess', test1), ('stopTest', test1)], events)