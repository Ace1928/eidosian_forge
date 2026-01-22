from openstack.cloud import _utils
from openstack import exceptions
def search_coe_clusters(self, name_or_id=None, filters=None):
    """Search COE cluster.

        :param name_or_id: cluster name or ID.
        :param filters: a dict containing additional filters to use.
        :param detail: a boolean to control if we need summarized or
            detailed output.

        :returns: A list of container infrastructure management ``Cluster``
            objects.
        :raises: :class:`~openstack.exceptions.SDKException` if something goes
            wrong during the OpenStack API call.
        """
    coe_clusters = self.list_coe_clusters()
    return _utils._filter_list(coe_clusters, name_or_id, filters)