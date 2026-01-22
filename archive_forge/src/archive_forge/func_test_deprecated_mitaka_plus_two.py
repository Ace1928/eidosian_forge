from unittest import mock
from oslotest import base as test_base
from testtools import matchers
from oslo_log import versionutils
@mock.patch('oslo_log.versionutils.report_deprecated_feature')
def test_deprecated_mitaka_plus_two(self, mock_reporter):

    @versionutils.deprecated(as_of=versionutils.deprecated.MITAKA, remove_in=+2)
    class OutdatedClass(object):
        pass
    obj = OutdatedClass()
    self.assertIsInstance(obj, OutdatedClass)
    self.assert_deprecated(mock_reporter, what='OutdatedClass()', as_of='Mitaka', remove_in='Ocata')