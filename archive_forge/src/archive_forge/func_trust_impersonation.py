import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@trust_impersonation.setter
def trust_impersonation(self, value):
    self.root.setdefault('OS-TRUST:trust', {})['impersonation'] = value