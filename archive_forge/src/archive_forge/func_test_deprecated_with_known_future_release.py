from unittest import mock
from oslotest import base as test_base
from testtools import matchers
from oslo_log import versionutils
@mock.patch('oslo_log.versionutils.report_deprecated_feature')
def test_deprecated_with_known_future_release(self, mock_reporter):

    @versionutils.deprecated(as_of=versionutils.deprecated.GRIZZLY, in_favor_of='different_stuff()')
    def do_outdated_stuff():
        return
    do_outdated_stuff()
    self.assert_deprecated(mock_reporter, what='do_outdated_stuff()', in_favor_of='different_stuff()', as_of='Grizzly', remove_in='Icehouse')