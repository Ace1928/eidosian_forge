import os
import os.path
import stat
import unittest
from fixtures import MockPatch, TempDir
from testtools import TestCase
from lazr.restfulclient.authorize.oauth import (
def test_empty_distro_name(self):
    self.useFixture(MockPatch('distro.name', return_value=''))
    self.useFixture(MockPatch('platform.system', return_value='BarOS'))
    self.useFixture(MockPatch('socket.gethostname', return_value='bar'))
    consumer = SystemWideConsumer('app name')
    self.assertEqual(consumer.key, 'System-wide: BarOS (bar)')