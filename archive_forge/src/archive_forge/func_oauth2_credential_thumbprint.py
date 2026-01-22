import functools
from keystoneauth1 import _utils as utils
from keystoneauth1.access import service_catalog
from keystoneauth1.access import service_providers
@_missingproperty
def oauth2_credential_thumbprint(self):
    return self._data['token']['oauth2_credential']['x5t#S256']