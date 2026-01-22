import re
from tempest.lib import exceptions
from glanceclient.tests.functional import base
def test_fake_action(self):
    self.assertRaises(exceptions.CommandFailed, self.glance, 'this-does-not-exist')