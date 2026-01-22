from functools import wraps
import logging
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import units
from glance_store import capabilities
from glance_store import exceptions
from glance_store.i18n import _
def get_store_location_class(self):
    """
        Returns the store location class that is used by this store.
        """
    if not self.store_location_class:
        class_name = '%s.StoreLocation' % self.__module__
        LOG.debug('Late loading location class %s', class_name)
        self.store_location_class = importutils.import_class(class_name)
    return self.store_location_class