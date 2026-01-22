from libcloud.common.aws import AWSJsonResponse, SignedAWSConnection
from libcloud.container.base import Container, ContainerImage, ContainerDriver, ContainerCluster
from libcloud.container.types import ContainerState
from libcloud.container.utils.docker import RegistryClient
def ex_start_task(self, task_arn, count=1):
    """
        Run a task definition and get the containers

        :param task_arn: The task ARN to Run
        :type  task_arn: ``str``

        :param count: The number of containers to start
        :type  count: ``int``

        :rtype: ``list`` of :class:`libcloud.container.base.Container`
        """
    request = None
    request = {'count': count, 'taskDefinition': task_arn}
    response = self.connection.request(ROOT, method='POST', data=json.dumps(request), headers=self._get_headers('RunTask')).object
    containers = []
    for task in response['tasks']:
        containers.extend(self._to_containers(task, task_arn))
    return containers