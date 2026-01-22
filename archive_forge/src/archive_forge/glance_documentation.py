from oslo_config import cfg
from oslo_utils import uuidutils
from glanceclient import client as gc
from glanceclient import exc
from heat.engine.clients import client_exception
from heat.engine.clients import client_plugin
from heat.engine.clients import os as os_client
from heat.engine import constraints
Return the image object for the specified image name/id.

        :param image_identifier: image name
        :returns: an image object with name/id :image_identifier:
        