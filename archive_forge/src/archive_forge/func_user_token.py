import itertools
from oslo_serialization import jsonutils
import webob
@property
def user_token(self):
    return self.headers.get('X-Auth-Token', self.headers.get('X-Storage-Token'))