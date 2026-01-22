import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.ec2containerservice import exceptions
def discover_poll_endpoint(self, container_instance=None):
    """
        This action is only used by the Amazon EC2 Container Service
        agent, and it is not intended for use outside of the agent.


        Returns an endpoint for the Amazon EC2 Container Service agent
        to poll for updates.

        :type container_instance: string
        :param container_instance: The container instance UUID or full Amazon
            Resource Name (ARN) of the container instance. The ARN contains the
            `arn:aws:ecs` namespace, followed by the region of the container
            instance, the AWS account ID of the container instance owner, the
            `container-instance` namespace, and then the container instance
            UUID. For example, arn:aws:ecs: region : aws_account_id :container-
            instance/ container_instance_UUID .

        """
    params = {}
    if container_instance is not None:
        params['containerInstance'] = container_instance
    return self._make_request(action='DiscoverPollEndpoint', verb='POST', path='/', params=params)