import os
import glob
from abc import abstractmethod
from unittest import mock
from typing import Any, Optional
import pytest
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase
def test_single_pushpop(self):
    q = self.queue()
    q.push(b'a')
    assert q.pop() == b'a'