import sys
import unittest
from unittest import TestCase
from libcloud.compute.types import (
def test_nodestate_fromstring(self):
    self.assertEqual(NodeState.fromstring('running'), NodeState.RUNNING)