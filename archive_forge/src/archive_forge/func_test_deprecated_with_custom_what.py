from unittest import mock
from oslotest import base as test_base
from testtools import matchers
from oslo_log import versionutils
@mock.patch('oslo_log.versionutils.report_deprecated_feature')
def test_deprecated_with_custom_what(self, mock_reporter):

    @versionutils.deprecated(as_of=versionutils.deprecated.GRIZZLY, what='v2.0 API', in_favor_of='v3 API')
    def do_outdated_stuff():
        return
    do_outdated_stuff()
    self.assert_deprecated(mock_reporter, what='v2.0 API', in_favor_of='v3 API', as_of='Grizzly', remove_in='Icehouse')