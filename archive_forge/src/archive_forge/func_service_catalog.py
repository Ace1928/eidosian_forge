import functools
from keystoneauth1 import _utils as utils
from keystoneauth1.access import service_catalog
from keystoneauth1.access import service_providers
@property
def service_catalog(self):
    if not self._service_catalog:
        self._service_catalog = self._service_catalog_class.from_token(self._data)
    return self._service_catalog