import base64
import functools
import operator
import time
import iso8601
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack.compute.v2._proxy import Proxy
from openstack.compute.v2 import quota_set as _qs
from openstack.compute.v2 import server as _server
from openstack import exceptions
from openstack import utils
def get_server_by_id(self, id):
    """Get a server by ID.

        :param id: ID of the server.

        :returns: A compute ``Server`` object if found, else None.
        """
    try:
        server = self.compute.get_server(id)
        return meta.add_server_interfaces(self, server)
    except exceptions.ResourceNotFound:
        return None