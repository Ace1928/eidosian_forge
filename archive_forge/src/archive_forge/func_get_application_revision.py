import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.codedeploy import exceptions
def get_application_revision(self, application_name, revision):
    """
        Gets information about an application revision.

        :type application_name: string
        :param application_name: The name of the application that corresponds
            to the revision.

        :type revision: dict
        :param revision: Information about the application revision to get,
            including the revision's type and its location.

        """
    params = {'applicationName': application_name, 'revision': revision}
    return self.make_request(action='GetApplicationRevision', body=json.dumps(params))