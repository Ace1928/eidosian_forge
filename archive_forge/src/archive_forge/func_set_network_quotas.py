from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def set_network_quotas(self, name_or_id, **kwargs):
    """Set a network quota in a project

        :param name_or_id: project name or id
        :param kwargs: key/value pairs of quota name and quota value

        :raises: :class:`~openstack.exceptions.SDKException` if the resource to
            set the quota does not exist.
        """
    proj = self.get_project(name_or_id)
    if not proj:
        raise exceptions.SDKException('project does not exist')
    self.network.update_quota(proj.id, **kwargs)