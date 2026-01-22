from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_replica_export_locations
def test_list_share_replica_export_locations(self):
    share_replica_id = '1234'
    cs.share_replica_export_locations.list(share_replica_id)
    cs.assert_called('GET', '/share-replicas/%s/export-locations' % share_replica_id)