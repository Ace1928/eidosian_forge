import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
def set_domain_scope(self, id=None, name=None):
    self.domain_id = id or uuid.uuid4().hex
    self.domain_name = name or uuid.uuid4().hex