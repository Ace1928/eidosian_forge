from libcloud.common.aws import AWSJsonResponse, SignedAWSConnection
from libcloud.container.base import Container, ContainerImage, ContainerDriver, ContainerCluster
from libcloud.container.types import ContainerState
from libcloud.container.utils.docker import RegistryClient
def ex_get_registry_client(self, repository_name):
    """
        Get a client for an ECR repository

        :param  repository_name: The unique name of the repository
        :type   repository_name: ``str``

        :return: a docker registry API client
        :rtype: :class:`libcloud.container.utils.docker.RegistryClient`
        """
    repository_id = self.ex_get_repository_id(repository_name)
    token = self.ex_get_repository_token(repository_id)
    host = self._get_ecr_host(repository_id)
    return RegistryClient(host=host, username='AWS', password=token)