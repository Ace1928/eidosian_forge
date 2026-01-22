import logging
import os
import urllib.parse
from oslo_config import cfg
from oslo_utils import excutils
from oslo_utils import netutils
from oslo_utils import units
import requests
from requests import adapters
from requests.packages.urllib3.util import retry
import glance_store
from glance_store import capabilities
from glance_store.common import utils
from glance_store import exceptions
from glance_store.i18n import _, _LE
from glance_store import location
def select_datastore(self, image_size):
    """Select a datastore with free space larger than image size."""
    for k, v in sorted(self.datastores.items(), reverse=True):
        max_ds = None
        max_fs = 0
        for ds in v:
            ds.freespace = self._get_freespace(ds)
            if ds.freespace > max_fs:
                max_ds = ds
                max_fs = ds.freespace
        if max_ds and max_ds.freespace >= image_size:
            return max_ds
    msg = _LE('No datastore found with enough free space to contain an image of size %d') % image_size
    LOG.error(msg)
    raise exceptions.StorageFull()