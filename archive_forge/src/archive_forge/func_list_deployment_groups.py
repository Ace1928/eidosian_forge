import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.codedeploy import exceptions
def list_deployment_groups(self, application_name, next_token=None):
    """
        Lists the deployment groups for an application registered
        within the AWS user account.

        :type application_name: string
        :param application_name: The name of an existing AWS CodeDeploy
            application within the AWS user account.

        :type next_token: string
        :param next_token: An identifier that was returned from the previous
            list deployment groups call, which can be used to return the next
            set of deployment groups in the list.

        """
    params = {'applicationName': application_name}
    if next_token is not None:
        params['nextToken'] = next_token
    return self.make_request(action='ListDeploymentGroups', body=json.dumps(params))