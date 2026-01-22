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
def get_cached_images(self):
    """
        Returns a list of records about cached images.
        """
    LOG.debug('Gathering cached image entries.')
    entries = []
    for path in get_all_regular_files(self.base_dir):
        image_id = os.path.basename(path)
        entry = {'image_id': image_id}
        file_info = os.stat(path)
        entry['last_modified'] = file_info[stat.ST_MTIME]
        entry['last_accessed'] = file_info[stat.ST_ATIME]
        entry['size'] = file_info[stat.ST_SIZE]
        entry['hits'] = self.get_hit_count(image_id)
        entries.append(entry)
    return entries