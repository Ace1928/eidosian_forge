import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@domain_name.setter
def domain_name(self, value):
    self.root.setdefault('domain', {})['name'] = value