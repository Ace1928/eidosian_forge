import itertools
from oslo_serialization import jsonutils
import webob
@service_token_valid.setter
def service_token_valid(self, value):
    self.headers[self._SERVICE_STATUS_HEADER] = self._confirmed(value)