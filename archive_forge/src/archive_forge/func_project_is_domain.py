import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@project_is_domain.setter
def project_is_domain(self, value):
    self.root['is_domain'] = value