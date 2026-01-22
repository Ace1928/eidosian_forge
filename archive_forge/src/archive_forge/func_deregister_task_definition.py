import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.ec2containerservice import exceptions
def deregister_task_definition(self, task_definition):
    """
        Deregisters the specified task definition. You will no longer
        be able to run tasks from this definition after
        deregistration.

        :type task_definition: string
        :param task_definition: The `family` and `revision` (
            `family:revision`) or full Amazon Resource Name (ARN) of the task
            definition that you want to deregister.

        """
    params = {'taskDefinition': task_definition}
    return self._make_request(action='DeregisterTaskDefinition', verb='POST', path='/', params=params)