import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.ec2containerservice import exceptions
def describe_container_instances(self, container_instances, cluster=None):
    """
        Describes Amazon EC2 Container Service container instances.
        Returns metadata about registered and remaining resources on
        each container instance requested.

        :type cluster: string
        :param cluster: The short name or full Amazon Resource Name (ARN) of
            the cluster that hosts the container instances you want to
            describe. If you do not specify a cluster, the default cluster is
            assumed.

        :type container_instances: list
        :param container_instances: A space-separated list of container
            instance UUIDs or full Amazon Resource Name (ARN) entries.

        """
    params = {}
    self.build_list_params(params, container_instances, 'containerInstances.member')
    if cluster is not None:
        params['cluster'] = cluster
    return self._make_request(action='DescribeContainerInstances', verb='POST', path='/', params=params)