import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def create_hsm_client_certificate(self, hsm_client_certificate_identifier):
    """
        Creates an HSM client certificate that an Amazon Redshift
        cluster will use to connect to the client's HSM in order to
        store and retrieve the keys used to encrypt the cluster
        databases.

        The command returns a public key, which you must store in the
        HSM. In addition to creating the HSM certificate, you must
        create an Amazon Redshift HSM configuration that provides a
        cluster the information needed to store and use encryption
        keys in the HSM. For more information, go to `Hardware
        Security Modules`_ in the Amazon Redshift Management Guide.

        :type hsm_client_certificate_identifier: string
        :param hsm_client_certificate_identifier: The identifier to be assigned
            to the new HSM client certificate that the cluster will use to
            connect to the HSM to use the database encryption keys.

        """
    params = {'HsmClientCertificateIdentifier': hsm_client_certificate_identifier}
    return self._make_request(action='CreateHsmClientCertificate', verb='POST', path='/', params=params)