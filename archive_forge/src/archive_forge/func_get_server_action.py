import warnings
from openstack.block_storage.v3 import volume as _volume
from openstack.compute.v2 import aggregate as _aggregate
from openstack.compute.v2 import availability_zone
from openstack.compute.v2 import extension
from openstack.compute.v2 import flavor as _flavor
from openstack.compute.v2 import hypervisor as _hypervisor
from openstack.compute.v2 import image as _image
from openstack.compute.v2 import keypair as _keypair
from openstack.compute.v2 import limits
from openstack.compute.v2 import migration as _migration
from openstack.compute.v2 import quota_set as _quota_set
from openstack.compute.v2 import server as _server
from openstack.compute.v2 import server_action as _server_action
from openstack.compute.v2 import server_diagnostics as _server_diagnostics
from openstack.compute.v2 import server_group as _server_group
from openstack.compute.v2 import server_interface as _server_interface
from openstack.compute.v2 import server_ip
from openstack.compute.v2 import server_migration as _server_migration
from openstack.compute.v2 import server_remote_console as _src
from openstack.compute.v2 import service as _service
from openstack.compute.v2 import usage as _usage
from openstack.compute.v2 import volume_attachment as _volume_attachment
from openstack import exceptions
from openstack.identity.v3 import project as _project
from openstack.network.v2 import security_group as _sg
from openstack import proxy
from openstack import resource
from openstack import utils
from openstack import warnings as os_warnings
def get_server_action(self, server_action, server, ignore_missing=True):
    """Get a single server action

        :param server_action: The value can be the ID of a server action or a
            :class:`~openstack.compute.v2.server_action.ServerAction` instance.
        :param server: This parameter need to be specified when ServerAction ID
            is given as value. It can be either the ID of a server or a
            :class:`~openstack.compute.v2.server.Server` instance that the
            action is associated with.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be raised when
            the server action does not exist. When set to ``True``, no
            exception will be set when attempting to retrieve a non-existent
            server action.

        :returns: One :class:`~openstack.compute.v2.server_action.ServerAction`
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            resource can be found.
        """
    server_id = self._get_uri_attribute(server_action, server, 'server_id')
    server_action = resource.Resource._get_id(server_action)
    return self._get(_server_action.ServerAction, server_id=server_id, request_id=server_action, ignore_missing=ignore_missing)