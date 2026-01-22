from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import (
from openstack import proxy
def get_cluster_certificate(self, cluster_certificate):
    """Get a single cluster_certificate

        :param cluster_certificate: The value can be the ID of a
            cluster_certificate or a
            :class:`~openstack.container_infrastructure_management.v1.cluster_certificate.ClusterCertificate`
            instance.

        :returns: One
            :class:`~openstack.container_infrastructure_management.v1.cluster_certificate.ClusterCertificate`
        :raises: :class:`~openstack.exceptions.ResourceNotFound`
            when no resource can be found.
        """
    return self._get(_cluster_cert.ClusterCertificate, cluster_certificate)