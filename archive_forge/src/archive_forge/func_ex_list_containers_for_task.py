from libcloud.common.aws import AWSJsonResponse, SignedAWSConnection
from libcloud.container.base import Container, ContainerImage, ContainerDriver, ContainerCluster
from libcloud.container.types import ContainerState
from libcloud.container.utils.docker import RegistryClient
def ex_list_containers_for_task(self, task_arns):
    """
        Get a list of containers by ID collection (ARN)

        :param task_arns: The list of ARNs
        :type  task_arns: ``list`` of ``str``

        :rtype: ``list`` of :class:`libcloud.container.base.Container`
        """
    describe_request = {'tasks': task_arns}
    descripe_response = self.connection.request(ROOT, method='POST', data=json.dumps(describe_request), headers=self._get_headers('DescribeTasks')).object
    containers = []
    for task in descripe_response['tasks']:
        containers.extend(self._to_containers(task, task['taskDefinitionArn']))
    return containers