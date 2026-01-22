import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def update_app(self, app_id, name=None, description=None, data_sources=None, type=None, app_source=None, domains=None, enable_ssl=None, ssl_configuration=None, attributes=None, environment=None):
    """
        Updates a specified app.

        **Required Permissions**: To use this action, an IAM user must
        have a Deploy or Manage permissions level for the stack, or an
        attached policy that explicitly grants permissions. For more
        information on user permissions, see `Managing User
        Permissions`_.

        :type app_id: string
        :param app_id: The app ID.

        :type name: string
        :param name: The app name.

        :type description: string
        :param description: A description of the app.

        :type data_sources: list
        :param data_sources: The app's data sources.

        :type type: string
        :param type: The app type.

        :type app_source: dict
        :param app_source: A `Source` object that specifies the app repository.

        :type domains: list
        :param domains: The app's virtual host settings, with multiple domains
            separated by commas. For example: `'www.example.com, example.com'`

        :type enable_ssl: boolean
        :param enable_ssl: Whether SSL is enabled for the app.

        :type ssl_configuration: dict
        :param ssl_configuration: An `SslConfiguration` object with the SSL
            configuration.

        :type attributes: map
        :param attributes: One or more user-defined key/value pairs to be added
            to the stack attributes.

        :type environment: list
        :param environment:
        An array of `EnvironmentVariable` objects that specify environment
            variables to be associated with the app. You can specify up to ten
            environment variables. After you deploy the app, these variables
            are defined on the associated app server instances.

        This parameter is supported only by Chef 11.10 stacks. If you have
            specified one or more environment variables, you cannot modify the
            stack's Chef version.

        """
    params = {'AppId': app_id}
    if name is not None:
        params['Name'] = name
    if description is not None:
        params['Description'] = description
    if data_sources is not None:
        params['DataSources'] = data_sources
    if type is not None:
        params['Type'] = type
    if app_source is not None:
        params['AppSource'] = app_source
    if domains is not None:
        params['Domains'] = domains
    if enable_ssl is not None:
        params['EnableSsl'] = enable_ssl
    if ssl_configuration is not None:
        params['SslConfiguration'] = ssl_configuration
    if attributes is not None:
        params['Attributes'] = attributes
    if environment is not None:
        params['Environment'] = environment
    return self.make_request(action='UpdateApp', body=json.dumps(params))