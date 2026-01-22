import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def delete_hsm_configuration(self, hsm_configuration_identifier):
    """
        Deletes the specified Amazon Redshift HSM configuration.

        :type hsm_configuration_identifier: string
        :param hsm_configuration_identifier: The identifier of the Amazon
            Redshift HSM configuration to be deleted.

        """
    params = {'HsmConfigurationIdentifier': hsm_configuration_identifier}
    return self._make_request(action='DeleteHsmConfiguration', verb='POST', path='/', params=params)