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
def evacuate_server(self, server, host=None, admin_pass=None, force=None):
    """Evacuates a server from a failed host to a new host.

        :param server: Either the ID of a server or a
            :class:`~openstack.compute.v2.server.Server` instance.
        :param host: An optional parameter specifying the name or ID of the
            host to which the server is evacuated.
        :param admin_pass: An optional parameter specifying the administrative
            password to access the evacuated or rebuilt server.
        :param force: Force an evacuation by not verifying the provided
            destination host by the scheduler. (New in API version
            2.29).
        :returns: None
        """
    server = self._get_resource(_server.Server, server)
    server.evacuate(self, host=host, admin_pass=admin_pass, force=force)