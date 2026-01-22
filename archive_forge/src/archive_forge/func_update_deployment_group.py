import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.codedeploy import exceptions
def update_deployment_group(self, application_name, current_deployment_group_name, new_deployment_group_name=None, deployment_config_name=None, ec_2_tag_filters=None, auto_scaling_groups=None, service_role_arn=None):
    """
        Changes information about an existing deployment group.

        :type application_name: string
        :param application_name: The application name corresponding to the
            deployment group to update.

        :type current_deployment_group_name: string
        :param current_deployment_group_name: The current name of the existing
            deployment group.

        :type new_deployment_group_name: string
        :param new_deployment_group_name: The new name of the deployment group,
            if you want to change it.

        :type deployment_config_name: string
        :param deployment_config_name: The replacement deployment configuration
            name to use, if you want to change it.

        :type ec_2_tag_filters: list
        :param ec_2_tag_filters: The replacement set of Amazon EC2 tags to
            filter on, if you want to change them.

        :type auto_scaling_groups: list
        :param auto_scaling_groups: The replacement list of Auto Scaling groups
            to be included in the deployment group, if you want to change them.

        :type service_role_arn: string
        :param service_role_arn: A replacement service role's ARN, if you want
            to change it.

        """
    params = {'applicationName': application_name, 'currentDeploymentGroupName': current_deployment_group_name}
    if new_deployment_group_name is not None:
        params['newDeploymentGroupName'] = new_deployment_group_name
    if deployment_config_name is not None:
        params['deploymentConfigName'] = deployment_config_name
    if ec_2_tag_filters is not None:
        params['ec2TagFilters'] = ec_2_tag_filters
    if auto_scaling_groups is not None:
        params['autoScalingGroups'] = auto_scaling_groups
    if service_role_arn is not None:
        params['serviceRoleArn'] = service_role_arn
    return self.make_request(action='UpdateDeploymentGroup', body=json.dumps(params))