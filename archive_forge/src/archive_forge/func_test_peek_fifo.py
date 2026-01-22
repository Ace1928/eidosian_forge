import os
import glob
from abc import abstractmethod
from unittest import mock
from typing import Any, Optional
import pytest
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase
def test_peek_fifo(self):
    q = self.queue()
    self.assertIsNone(q.peek())
    q.push(b'a')
    q.push(b'b')
    q.push(b'c')
    self.assertEqual(q.peek(), b'a')
    self.assertEqual(q.peek(), b'a')
    self.assertEqual(q.pop(), b'a')
    self.assertEqual(q.peek(), b'b')
    self.assertEqual(q.peek(), b'b')
    self.assertEqual(q.pop(), b'b')
    self.assertEqual(q.peek(), b'c')
    self.assertEqual(q.peek(), b'c')
    self.assertEqual(q.pop(), b'c')
    self.assertIsNone(q.peek())