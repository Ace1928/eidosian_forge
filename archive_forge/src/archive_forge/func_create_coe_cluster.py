from openstack.cloud import _utils
from openstack import exceptions
def create_coe_cluster(self, name, cluster_template_id, **kwargs):
    """Create a COE cluster based on given cluster template.

        :param string name: Name of the cluster.
        :param string cluster_template_id: ID of the cluster template to use.
        :param dict kwargs: Any other arguments to pass in.

        :returns: The created container infrastructure management ``Cluster``
            object.
        :raises: :class:`~openstack.exceptions.SDKException` if something goes
            wrong during the OpenStack API call
        """
    cluster = self.container_infrastructure_management.create_cluster(name=name, cluster_template_id=cluster_template_id, **kwargs)
    return cluster