import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
def set_bind(self, name, data):
    self._token.setdefault('bind', {})[name] = data