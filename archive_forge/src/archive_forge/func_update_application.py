import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.codedeploy import exceptions
def update_application(self, application_name=None, new_application_name=None):
    """
        Changes an existing application's name.

        :type application_name: string
        :param application_name: The current name of the application that you
            want to change.

        :type new_application_name: string
        :param new_application_name: The new name that you want to change the
            application to.

        """
    params = {}
    if application_name is not None:
        params['applicationName'] = application_name
    if new_application_name is not None:
        params['newApplicationName'] = new_application_name
    return self.make_request(action='UpdateApplication', body=json.dumps(params))