import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_from_imports():
    from six.moves.queue import Queue
    assert isinstance(Queue, six.class_types)
    from six.moves.configparser import ConfigParser
    assert isinstance(ConfigParser, six.class_types)