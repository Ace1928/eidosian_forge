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
def test_tags_tests(self):
    result = ExtendedTestResult()
    tagger = Tagger(result, {'foo'}, {'bar'})
    test1, test2 = (self, make_test())
    tagger.startTest(test1)
    tagger.addSuccess(test1)
    tagger.stopTest(test1)
    tagger.startTest(test2)
    tagger.addSuccess(test2)
    tagger.stopTest(test2)
    self.assertEqual([('startTest', test1), ('tags', {'foo'}, {'bar'}), ('addSuccess', test1), ('stopTest', test1), ('startTest', test2), ('tags', {'foo'}, {'bar'}), ('addSuccess', test2), ('stopTest', test2)], result._events)