from keystoneauth1 import discover
from keystoneauth1.identity.generic import base
from keystoneauth1.identity import v2
from keystoneauth1.identity import v3
@user_domain_name.setter
def user_domain_name(self, value):
    self._user_domain_name = value