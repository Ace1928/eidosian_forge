import os
import glob
from abc import abstractmethod
from unittest import mock
from typing import Any, Optional
import pytest
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase
def test_binary_element(self):
    elem = b'\x80\x02}q\x01(U\x04bodyq\x02U\x00U\t_encodingq\x03U\x05utf-8q\x04U\x07cookiesq\x05}q\x06U\x04metaq\x07}q\x08U\x07headersq\t}U\x03urlq\nX\x15\x00\x00\x00file:///tmp/tmphDJYsgU\x0bdont_filterq\x0b\x89U\x08priorityq\x0cK\x00U\x08callbackq\rNU\x06methodq\x0eU\x03GETq\x0fU\x07errbackq\x10Nu.'
    q = self.queue()
    q.push(elem)
    assert q.pop() == elem