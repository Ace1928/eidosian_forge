import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_sanitize_keys(self):
    lowered = [k.lower() for k in strutils._SANITIZE_KEYS]
    message = 'The _SANITIZE_KEYS must all be lowercase.'
    self.assertEqual(strutils._SANITIZE_KEYS, lowered, message)