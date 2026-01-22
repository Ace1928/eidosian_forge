import queue
import threading
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
import webob.exc
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import wsgi
import glance.db
import glance.gateway
from glance.i18n import _
from glance import image_cache
import glance.notifier
def get_cache_state(self, req):
    """
        GET /cache/ - Get currently cached and queued images

        Returns dict of cached and queued images
        """
    self._enforce(req, new_policy='cache_list')
    return dict(cached_images=self.cache.get_cached_images(), queued_images=self.cache.get_queued_images())