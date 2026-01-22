from libcloud.common.aws import AWSJsonResponse, SignedAWSConnection
from libcloud.container.base import Container, ContainerImage, ContainerDriver, ContainerCluster
from libcloud.container.types import ContainerState
from libcloud.container.utils.docker import RegistryClient
def ex_get_repository_id(self, repository_name):
    """
        Get the ID of a repository

        :param  repository_name: The unique name of the repository
        :type   repository_name: ``str``

        :return: The repository ID
        :rtype: ``str``
        """
    request = {'repositoryNames': [repository_name]}
    list_response = self.ecr_connection.request(ROOT, method='POST', data=json.dumps(request), headers=self._get_ecr_headers('DescribeRepositories')).object
    repository_id = list_response['repositories'][0]['registryId']
    return repository_id