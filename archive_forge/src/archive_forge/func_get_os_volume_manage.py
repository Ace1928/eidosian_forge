from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def get_os_volume_manage(self, **kw):
    vol_id = 'volume-ffffffff-0000-ffff-0000-ffffffffffff'
    vols = [{'size': 4, 'safe_to_manage': False, 'actual_size': 4.0, 'reference': {'source-name': vol_id}}, {'size': 5, 'safe_to_manage': True, 'actual_size': 4.3, 'reference': {'source-name': 'myvol'}}]
    return (200, {}, {'manageable-volumes': vols})