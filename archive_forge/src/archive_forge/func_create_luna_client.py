import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudhsm import exceptions
def create_luna_client(self, certificate, label=None):
    """
        Creates an HSM client.

        :type label: string
        :param label: The label for the client.

        :type certificate: string
        :param certificate: The contents of a Base64-Encoded X.509 v3
            certificate to be installed on the HSMs used by this client.

        """
    params = {'Certificate': certificate}
    if label is not None:
        params['Label'] = label
    return self.make_request(action='CreateLunaClient', body=json.dumps(params))