from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_unicode_property(self):
    """
        Sample data can be created by setting L{FakeLaunchpad} attributes with
        dicts that represent objects.  Plain string values can be represented
        as C{unicode} strings.
        """
    self.launchpad.me = dict(name=u'foo')
    self.assertEqual(u'foo', self.launchpad.me.name)