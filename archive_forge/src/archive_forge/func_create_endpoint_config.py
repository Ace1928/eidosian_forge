from datetime import datetime
from .. import errors
from .. import utils
from ..constants import DEFAULT_DATA_CHUNK_SIZE
from ..types import CancellableStream
from ..types import ContainerConfig
from ..types import EndpointConfig
from ..types import HostConfig
from ..types import NetworkingConfig
def create_endpoint_config(self, *args, **kwargs):
    """
        Create an endpoint config dictionary to be used with
        :py:meth:`create_networking_config`.

        Args:
            aliases (:py:class:`list`): A list of aliases for this endpoint.
                Names in that list can be used within the network to reach the
                container. Defaults to ``None``.
            links (dict): Mapping of links for this endpoint using the
                ``{'container': 'alias'}`` format. The alias is optional.
                Containers declared in this dict will be linked to this
                container using the provided alias. Defaults to ``None``.
            ipv4_address (str): The IP address of this container on the
                network, using the IPv4 protocol. Defaults to ``None``.
            ipv6_address (str): The IP address of this container on the
                network, using the IPv6 protocol. Defaults to ``None``.
            link_local_ips (:py:class:`list`): A list of link-local (IPv4/IPv6)
                addresses.
            driver_opt (dict): A dictionary of options to provide to the
                network driver. Defaults to ``None``.

        Returns:
            (dict) An endpoint config.

        Example:

            >>> endpoint_config = client.api.create_endpoint_config(
                aliases=['web', 'app'],
                links={'app_db': 'db', 'another': None},
                ipv4_address='132.65.0.123'
            )

        """
    return EndpointConfig(self._version, *args, **kwargs)