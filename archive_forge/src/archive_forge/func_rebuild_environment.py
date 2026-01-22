import boto
import boto.jsonresponse
from boto.compat import json
from boto.regioninfo import RegionInfo
from boto.connection import AWSQueryConnection
def rebuild_environment(self, environment_id=None, environment_name=None):
    """
        Deletes and recreates all of the AWS resources (for example:
        the Auto Scaling group, load balancer, etc.) for a specified
        environment and forces a restart.

        :type environment_id: string
        :param environment_id: The ID of the environment to rebuild.
            Condition: You must specify either this or an EnvironmentName, or
            both.  If you do not specify either, AWS Elastic Beanstalk returns
            MissingRequiredParameter error.

        :type environment_name: string
        :param environment_name: The name of the environment to rebuild.
            Condition: You must specify either this or an EnvironmentId, or
            both.  If you do not specify either, AWS Elastic Beanstalk returns
            MissingRequiredParameter error.

        :raises InvalidParameterValue: If environment_name doesn't refer to a currently active environment
        :raises: InsufficientPrivilegesException
        """
    params = {}
    if environment_id:
        params['EnvironmentId'] = environment_id
    if environment_name:
        params['EnvironmentName'] = environment_name
    return self._get_response('RebuildEnvironment', params)