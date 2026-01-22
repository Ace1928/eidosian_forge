import os
import glob
from abc import abstractmethod
from unittest import mock
from typing import Any, Optional
import pytest
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase
def test_peek_one_element(self):
    q = self.queue()
    self.assertIsNone(q.peek())
    q.push(b'a')
    self.assertEqual(q.peek(), b'a')
    self.assertEqual(q.pop(), b'a')
    self.assertIsNone(q.peek())