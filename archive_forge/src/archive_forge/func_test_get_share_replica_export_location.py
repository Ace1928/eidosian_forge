from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_replica_export_locations
def test_get_share_replica_export_location(self):
    share_replica_id = '1234'
    el_uuid = 'fake_el_uuid'
    cs.share_replica_export_locations.get(share_replica_id, el_uuid)
    url = '/share-replicas/%(share_replica_id)s/export-locations/%(el_uuid)s'
    payload = {'share_replica_id': share_replica_id, 'el_uuid': el_uuid}
    cs.assert_called('GET', url % payload)