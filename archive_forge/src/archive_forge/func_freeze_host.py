from cinderclient import api_versions
from cinderclient import base
def freeze_host(self, host):
    """Freeze the service specified by hostname."""
    body = {'host': host}
    return self._update('/os-services/freeze', body)