import sys
import unittest
from unittest import TestCase
from libcloud.compute.types import (
def test_storagevolumestate_tostring(self):
    self.assertEqual(StorageVolumeState.tostring(StorageVolumeState.AVAILABLE), 'AVAILABLE')