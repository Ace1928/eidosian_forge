from oslo_config import cfg
from oslo_utils import uuidutils
from glanceclient import client as gc
from glanceclient import exc
from heat.engine.clients import client_exception
from heat.engine.clients import client_plugin
from heat.engine.clients import os as os_client
from heat.engine import constraints
def find_image_by_name_or_id(self, image_identifier):
    """Return the ID for the specified image name or identifier.

        :param image_identifier: image name or a UUID-like identifier
        :returns: the id of the requested :image_identifier:
        """
    return self._find_image_id(self.context.tenant_id, image_identifier)