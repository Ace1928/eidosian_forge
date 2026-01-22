from designateclient.v2.base import V2Controller
from designateclient.v2 import utils as v2_utils
def get_import_record(self, zone_import_id):
    return self._get(f'/zones/tasks/imports/{zone_import_id}')