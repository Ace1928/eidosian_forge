from .api.client import APIClient
from .constants import (DEFAULT_TIMEOUT_SECONDS, DEFAULT_MAX_POOL_SIZE)
from .models.configs import ConfigCollection
from .models.containers import ContainerCollection
from .models.images import ImageCollection
from .models.networks import NetworkCollection
from .models.nodes import NodeCollection
from .models.plugins import PluginCollection
from .models.secrets import SecretCollection
from .models.services import ServiceCollection
from .models.swarm import Swarm
from .models.volumes import VolumeCollection
from .utils import kwargs_from_env
from_env = DockerClient.from_env
@classmethod
def from_env(cls, **kwargs):
    """
        Return a client configured from environment variables.

        The environment variables used are the same as those used by the
        Docker command-line client. They are:

        .. envvar:: DOCKER_HOST

            The URL to the Docker host.

        .. envvar:: DOCKER_TLS_VERIFY

            Verify the host against a CA certificate.

        .. envvar:: DOCKER_CERT_PATH

            A path to a directory containing TLS certificates to use when
            connecting to the Docker host.

        Args:
            version (str): The version of the API to use. Set to ``auto`` to
                automatically detect the server's version. Default: ``auto``
            timeout (int): Default timeout for API calls, in seconds.
            max_pool_size (int): The maximum number of connections
                to save in the pool.
            ssl_version (int): A valid `SSL version`_.
            assert_hostname (bool): Verify the hostname of the server.
            environment (dict): The environment to read environment variables
                from. Default: the value of ``os.environ``
            credstore_env (dict): Override environment variables when calling
                the credential store process.
            use_ssh_client (bool): If set to `True`, an ssh connection is
                made via shelling out to the ssh client. Ensure the ssh
                client is installed and configured on the host.

        Example:

            >>> import docker
            >>> client = docker.from_env()

        .. _`SSL version`:
            https://docs.python.org/3.5/library/ssl.html#ssl.PROTOCOL_TLSv1
        """
    timeout = kwargs.pop('timeout', DEFAULT_TIMEOUT_SECONDS)
    max_pool_size = kwargs.pop('max_pool_size', DEFAULT_MAX_POOL_SIZE)
    version = kwargs.pop('version', None)
    use_ssh_client = kwargs.pop('use_ssh_client', False)
    return cls(timeout=timeout, max_pool_size=max_pool_size, version=version, use_ssh_client=use_ssh_client, **kwargs_from_env(**kwargs))