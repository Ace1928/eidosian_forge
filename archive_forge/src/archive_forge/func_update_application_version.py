import boto
import boto.jsonresponse
from boto.compat import json
from boto.regioninfo import RegionInfo
from boto.connection import AWSQueryConnection
def update_application_version(self, application_name, version_label, description=None):
    """Updates the application version to have the properties.

        :type application_name: string
        :param application_name: The name of the application associated with
            this version.  If no application is found with this name,
            UpdateApplication returns an InvalidParameterValue error.

        :type version_label: string
        :param version_label: The name of the version to update. If no
            application version is found with this label, UpdateApplication
            returns an InvalidParameterValue error.

        :type description: string
        :param description: A new description for this release.
        """
    params = {'ApplicationName': application_name, 'VersionLabel': version_label}
    if description:
        params['Description'] = description
    return self._get_response('UpdateApplicationVersion', params)