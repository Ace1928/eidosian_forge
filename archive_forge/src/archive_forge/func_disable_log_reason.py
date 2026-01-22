from cinderclient import api_versions
from cinderclient import base
def disable_log_reason(self, host, binary, reason):
    """Disable the service with reason."""
    body = {'host': host, 'binary': binary, 'disabled_reason': reason}
    result = self._update('/os-services/disable-log-reason', body)
    return self.resource_class(self, result, resp=result.request_ids)