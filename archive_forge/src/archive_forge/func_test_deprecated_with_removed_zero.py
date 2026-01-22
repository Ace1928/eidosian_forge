from unittest import mock
from oslotest import base as test_base
from testtools import matchers
from oslo_log import versionutils
@mock.patch('oslo_log.versionutils.report_deprecated_feature')
def test_deprecated_with_removed_zero(self, mock_reporter):

    @versionutils.deprecated(as_of=versionutils.deprecated.GRIZZLY, remove_in=0)
    def do_outdated_stuff():
        return
    do_outdated_stuff()
    self.assert_deprecated(mock_reporter, no_removal=True, what='do_outdated_stuff()', as_of='Grizzly', remove_in='Grizzly')