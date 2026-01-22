from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_instance_export_locations
def test_list_of_export_locations(self):
    share_instance_id = '1234'
    cs.share_instance_export_locations.list(share_instance_id, search_opts=None)
    cs.assert_called('GET', '/share_instances/%s/export_locations' % share_instance_id)