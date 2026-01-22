import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@audit_chain_id.setter
def audit_chain_id(self, value):
    self._token['audit_ids'] = [self.audit_id, value]