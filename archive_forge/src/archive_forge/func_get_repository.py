from base64 import b64encode
from libcloud.common.base import Connection, JsonResponse
from libcloud.container.base import ContainerImage
def get_repository(self, repository_name, namespace='library'):
    """
        Get the information about a specific repository

        :param  repository_name: The name of the repository e.g. 'ubuntu'
        :type   repository_name: ``str``

        :param  namespace: (optional) The docker namespace
        :type   namespace: ``str``

        :return: The details of the repository
        :rtype: ``object``
        """
    path = '/v2/repositories/{}/{}/'.format(namespace, repository_name)
    response = self.connection.request(path)
    return response.object