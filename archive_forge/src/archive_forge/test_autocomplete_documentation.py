import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
This is a mock numpy object that raises an error when there is an attempt
    to convert it to a boolean.