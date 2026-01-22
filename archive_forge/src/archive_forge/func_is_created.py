from oslo_config import cfg
from oslo_log import log as logging
import webob.exc
from glance.api import policy
from glance.common import exception
from glance.i18n import _
@property
def is_created(self):
    """Signal whether the image actually exists or not.

        False if the image is only being proposed by a create operation,
        True if it has already been created.
        """
    return not isinstance(self._image, dict)