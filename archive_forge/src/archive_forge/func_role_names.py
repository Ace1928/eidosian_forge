import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@property
def role_names(self):
    return [r['name'] for r in self.root.get('roles', [])]