from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import (
from openstack.container_infrastructure_management.v1 import (
from openstack import proxy
def get_cluster_template(self, cluster_template):
    """Get a single cluster_template

        :param cluster_template: The value can be the ID of a cluster_template
            or a
            :class:`~openstack.container_infrastructure_management.v1.cluster_template.ClusterTemplate`
            instance.

        :returns: One
            :class:`~openstack.container_infrastructure_management.v1.cluster_template.ClusterTemplate`
        :raises: :class:`~openstack.exceptions.ResourceNotFound`
            when no resource can be found.
        """
    return self._get(_cluster_template.ClusterTemplate, cluster_template)