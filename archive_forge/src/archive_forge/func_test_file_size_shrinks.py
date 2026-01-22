import os
import glob
from abc import abstractmethod
from unittest import mock
from typing import Any, Optional
import pytest
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase
def test_file_size_shrinks(self):
    """Test size of queue file shrinks when popping items"""
    q = self.queue()
    q.push(b'a')
    q.push(b'b')
    q.close()
    size = os.path.getsize(self.qpath)
    q = self.queue()
    q.pop()
    q.close()
    assert os.path.getsize(self.qpath), size