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
def delete_all_queued_images(self):
    """
        Removes all queued image files and any attributes about the images
        """
    files = [f for f in get_all_regular_files(self.queue_dir)]
    for file in files:
        fileutils.delete_if_exists(file)
    return len(files)