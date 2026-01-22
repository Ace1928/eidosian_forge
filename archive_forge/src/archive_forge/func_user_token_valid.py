import itertools
from oslo_serialization import jsonutils
import webob
@user_token_valid.setter
def user_token_valid(self, value):
    self.headers[self._USER_STATUS_HEADER] = self._confirmed(value)