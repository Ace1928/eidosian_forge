from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def get_manageable_volumes(self, **kw):
    vol_id = 'volume-ffffffff-0000-ffff-0000-ffffffffffff'
    vols = [{'size': 4, 'safe_to_manage': False, 'actual_size': 4.0, 'reference': {'source-name': vol_id}}, {'size': 5, 'safe_to_manage': True, 'actual_size': 4.3, 'reference': {'source-name': 'myvol'}}]
    return (200, {}, {'manageable-volumes': vols})