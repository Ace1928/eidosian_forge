from oslo_utils import uuidutils
from designateclient.v2.base import V2Controller
from designateclient.v2 import utils as v2_utils
def list_all_zones(self, criterion=None, marker=None, limit=None):
    url = self.build_url('/recordsets', criterion, marker, limit)
    return self._get(url, response_key='recordsets')