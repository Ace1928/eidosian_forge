from openstack.cloud import _utils
from openstack import exceptions
def sign_coe_cluster_certificate(self, cluster_id, csr):
    """Sign client key and generate the CA certificate for a cluster

        :param cluster_id: UUID of the cluster.
        :param csr: Certificate Signing Request (CSR) for authenticating
            client key.The CSR will be used by Magnum to generate a signed
            certificate that client will use to communicate with the cluster.

        :returns: a dict representing the signed certs.
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    return self.container_infrastructure_management.create_cluster_certificate(cluster_uuid=cluster_id, csr=csr)