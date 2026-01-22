import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def delete_server_cert(self, cert_name):
    """
        Delete the specified server certificate.

        :type cert_name: string
        :param cert_name: The name of the server certificate you want
            to delete.

        """
    params = {'ServerCertificateName': cert_name}
    return self.get_response('DeleteServerCertificate', params)