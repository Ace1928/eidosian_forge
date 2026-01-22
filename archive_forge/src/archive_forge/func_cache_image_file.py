from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import importutils
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance.common import exception
from glance.common import utils
from glance.i18n import _, _LE, _LI, _LW
def cache_image_file(self, image_id, image_file):
    """
        Cache an image file.

        :param image_id: Image ID
        :param image_file: Image file to cache

        :returns: True if image file was cached, False otherwise
        """
    CHUNKSIZE = 64 * units.Mi
    return self.cache_image_iter(image_id, utils.chunkiter(image_file, CHUNKSIZE))