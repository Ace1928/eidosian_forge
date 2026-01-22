from cinderclient.apiclient import base as common_base
from cinderclient import base
def list_manageable(self, host, detailed=True, marker=None, limit=None, offset=None, sort=None):
    return self.manager.list_manageable(host, detailed=detailed, marker=marker, limit=limit, offset=offset, sort=sort)