import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@trust_id.setter
def trust_id(self, value):
    self.root.setdefault('trust', {})['id'] = value