import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.codedeploy import exceptions
def delete_deployment_config(self, deployment_config_name):
    """
        Deletes a deployment configuration.

        A deployment configuration cannot be deleted if it is
        currently in use. Also, predefined configurations cannot be
        deleted.

        :type deployment_config_name: string
        :param deployment_config_name: The name of an existing deployment
            configuration within the AWS user account.

        """
    params = {'deploymentConfigName': deployment_config_name}
    return self.make_request(action='DeleteDeploymentConfig', body=json.dumps(params))