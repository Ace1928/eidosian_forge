import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
@pytest.fixture()
def self():
    return unittest.TestCase()