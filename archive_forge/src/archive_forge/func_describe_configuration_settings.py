import boto
import boto.jsonresponse
from boto.compat import json
from boto.regioninfo import RegionInfo
from boto.connection import AWSQueryConnection
def describe_configuration_settings(self, application_name, template_name=None, environment_name=None):
    """
        Returns a description of the settings for the specified
        configuration set, that is, either a configuration template or
        the configuration set associated with a running environment.
        When describing the settings for the configuration set
        associated with a running environment, it is possible to receive
        two sets of setting descriptions. One is the deployed
        configuration set, and the other is a draft configuration of an
        environment that is either in the process of deployment or that
        failed to deploy.

        :type application_name: string
        :param application_name: The application for the environment or
            configuration template.

        :type template_name: string
        :param template_name: The name of the configuration template to
            describe.  Conditional: You must specify either this parameter or
            an EnvironmentName, but not both. If you specify both, AWS Elastic
            Beanstalk returns an InvalidParameterCombination error.  If you do
            not specify either, AWS Elastic Beanstalk returns a
            MissingRequiredParameter error.

        :type environment_name: string
        :param environment_name: The name of the environment to describe.
            Condition: You must specify either this or a TemplateName, but not
            both. If you specify both, AWS Elastic Beanstalk returns an
            InvalidParameterCombination error. If you do not specify either,
            AWS Elastic Beanstalk returns MissingRequiredParameter error.
        """
    params = {'ApplicationName': application_name}
    if template_name:
        params['TemplateName'] = template_name
    if environment_name:
        params['EnvironmentName'] = environment_name
    return self._get_response('DescribeConfigurationSettings', params)