from designateclient.v2.base import V2Controller
from designateclient.v2 import utils as v2_utils
def pool_move(self, zone, values):
    zone = v2_utils.resolve_by_name(self.list, zone)
    url = self.build_url('/zones/%s/tasks/pool_move' % zone)
    return self._post(url, data=values)