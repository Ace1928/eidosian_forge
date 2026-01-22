import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@property
def service_providers(self):
    return self.root.get('service_providers')