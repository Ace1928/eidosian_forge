import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudhsm import exceptions
def modify_luna_client(self, client_arn, certificate):
    """
        Modifies the certificate used by the client.

        This action can potentially start a workflow to install the
        new certificate on the client's HSMs.

        :type client_arn: string
        :param client_arn: The ARN of the client.

        :type certificate: string
        :param certificate: The new certificate for the client.

        """
    params = {'ClientArn': client_arn, 'Certificate': certificate}
    return self.make_request(action='ModifyLunaClient', body=json.dumps(params))