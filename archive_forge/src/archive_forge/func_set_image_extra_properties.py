import copy
import functools
import json
import os
import urllib.request
import glance_store as store_api
from glance_store import backend
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import timeutils
from oslo_utils import units
import taskflow
from taskflow.patterns import linear_flow as lf
from taskflow import retry
from taskflow import task
from glance.api import common as api_common
import glance.async_.flows._internal_plugins as internal_plugins
import glance.async_.flows.plugins as import_plugins
from glance.async_ import utils
from glance.common import exception
from glance.common.scripts.image_import import main as image_import
from glance.common.scripts import utils as script_utils
from glance.common import store_utils
from glance.i18n import _, _LE, _LI
from glance.quota import keystone as ks_quota
def set_image_extra_properties(self, properties):
    """Merge values into image extra_properties.

        This allows a plugin to set additional properties on the image,
        as long as those are outside the reserved namespace. Any keys
        in the internal namespace will be dropped (and logged).

        :param properties: A dict of properties to be merged in
        """
    for key, value in properties.items():
        if key.startswith(api_common.GLANCE_RESERVED_NS):
            LOG.warning('Dropping %(key)s=%(val)s during metadata injection for %(image)s', {'key': key, 'val': value, 'image': self.image_id})
        else:
            self._image.extra_properties[key] = value