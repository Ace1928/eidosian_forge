import os
import glob
from abc import abstractmethod
from unittest import mock
from typing import Any, Optional
import pytest
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase
def test_push_pop2(self):
    """Test interleaved push and pops"""
    q = self.queue()
    q.push(b'a')
    q.push(b'b')
    q.push(b'c')
    q.push(b'd')
    self.assertEqual(q.pop(), b'd')
    self.assertEqual(q.pop(), b'c')
    q.push(b'e')
    self.assertEqual(q.pop(), b'e')
    self.assertEqual(q.pop(), b'b')
    self.assertEqual(q.pop(), b'a')