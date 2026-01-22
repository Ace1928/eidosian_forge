import os
import glob
from abc import abstractmethod
from unittest import mock
from typing import Any, Optional
import pytest
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase
def test_not_szhdr(self):
    q = self.queue()
    q.push(b'something')
    empty_file = open(self.tempfilename(), 'w+')
    with mock.patch.object(q, 'tailf', empty_file):
        assert q.peek() is None
        assert q.pop() is None
    empty_file.close()