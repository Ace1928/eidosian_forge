import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def create_hsm_configuration(self, hsm_configuration_identifier, description, hsm_ip_address, hsm_partition_name, hsm_partition_password, hsm_server_public_certificate):
    """
        Creates an HSM configuration that contains the information
        required by an Amazon Redshift cluster to store and use
        database encryption keys in a Hardware Security Module (HSM).
        After creating the HSM configuration, you can specify it as a
        parameter when creating a cluster. The cluster will then store
        its encryption keys in the HSM.

        In addition to creating an HSM configuration, you must also
        create an HSM client certificate. For more information, go to
        `Hardware Security Modules`_ in the Amazon Redshift Management
        Guide.

        :type hsm_configuration_identifier: string
        :param hsm_configuration_identifier: The identifier to be assigned to
            the new Amazon Redshift HSM configuration.

        :type description: string
        :param description: A text description of the HSM configuration to be
            created.

        :type hsm_ip_address: string
        :param hsm_ip_address: The IP address that the Amazon Redshift cluster
            must use to access the HSM.

        :type hsm_partition_name: string
        :param hsm_partition_name: The name of the partition in the HSM where
            the Amazon Redshift clusters will store their database encryption
            keys.

        :type hsm_partition_password: string
        :param hsm_partition_password: The password required to access the HSM
            partition.

        :type hsm_server_public_certificate: string
        :param hsm_server_public_certificate: The HSMs public certificate file.
            When using Cloud HSM, the file name is server.pem.

        """
    params = {'HsmConfigurationIdentifier': hsm_configuration_identifier, 'Description': description, 'HsmIpAddress': hsm_ip_address, 'HsmPartitionName': hsm_partition_name, 'HsmPartitionPassword': hsm_partition_password, 'HsmServerPublicCertificate': hsm_server_public_certificate}
    return self._make_request(action='CreateHsmConfiguration', verb='POST', path='/', params=params)