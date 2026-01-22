import itertools
from oslo_serialization import jsonutils
import webob
@token_auth.setter
def token_auth(self, v):
    self.environ[self._TOKEN_AUTH] = v