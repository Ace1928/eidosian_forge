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
def delete_all_cached_images(self):
    """
        Removes all cached image files and any attributes about the images
        """
    deleted = 0
    for path in get_all_regular_files(self.base_dir):
        delete_cached_file(path)
        deleted += 1
    return deleted