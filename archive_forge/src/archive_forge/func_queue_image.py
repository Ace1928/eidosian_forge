from contextlib import contextmanager
import errno
import os
import stat
import time
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import fileutils
import xattr
from glance.common import exception
from glance.i18n import _, _LI
from glance.image_cache.drivers import base
def queue_image(self, image_id):
    """
        This adds a image to be cache to the queue.

        If the image already exists in the queue or has already been
        cached, we return False, True otherwise

        :param image_id: Image ID
        """
    if self.is_cached(image_id):
        LOG.info(_LI("Not queueing image '%s'. Already cached."), image_id)
        return False
    if self.is_being_cached(image_id):
        LOG.info(_LI("Not queueing image '%s'. Already being written to cache"), image_id)
        return False
    if self.is_queued(image_id):
        LOG.info(_LI("Not queueing image '%s'. Already queued."), image_id)
        return False
    path = self.get_image_filepath(image_id, 'queue')
    LOG.debug("Queueing image '%s'.", image_id)
    with open(path, 'w'):
        pass
    return True