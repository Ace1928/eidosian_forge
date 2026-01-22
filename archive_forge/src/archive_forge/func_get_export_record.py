from designateclient.v2.base import V2Controller
from designateclient.v2 import utils as v2_utils
def get_export_record(self, zone_export_id):
    return self._get(f'/zones/tasks/exports/{zone_export_id}')