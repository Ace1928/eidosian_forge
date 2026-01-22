import boto
import boto.jsonresponse
from boto.compat import json
from boto.regioninfo import RegionInfo
from boto.connection import AWSQueryConnection
def describe_configuration_options(self, application_name=None, template_name=None, environment_name=None, solution_stack_name=None, options=None):
    """Describes configuration options used in a template or environment.

        Describes the configuration options that are used in a
        particular configuration template or environment, or that a
        specified solution stack defines. The description includes the
        values the options, their default values, and an indication of
        the required action on a running environment if an option value
        is changed.

        :type application_name: string
        :param application_name: The name of the application associated with
            the configuration template or environment. Only needed if you want
            to describe the configuration options associated with either the
            configuration template or environment.

        :type template_name: string
        :param template_name: The name of the configuration template whose
            configuration options you want to describe.

        :type environment_name: string
        :param environment_name: The name of the environment whose
            configuration options you want to describe.

        :type solution_stack_name: string
        :param solution_stack_name: The name of the solution stack whose
            configuration options you want to describe.

        :type options: list
        :param options: If specified, restricts the descriptions to only
            the specified options.
        """
    params = {}
    if application_name:
        params['ApplicationName'] = application_name
    if template_name:
        params['TemplateName'] = template_name
    if environment_name:
        params['EnvironmentName'] = environment_name
    if solution_stack_name:
        params['SolutionStackName'] = solution_stack_name
    if options:
        self.build_list_params(params, options, 'Options.member')
    return self._get_response('DescribeConfigurationOptions', params)