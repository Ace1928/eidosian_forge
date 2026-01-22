import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@tenant_id.setter
def tenant_id(self, value):
    self._token.setdefault('tenant', {})['id'] = value