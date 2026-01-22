from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def delete_network_quotas(self, name_or_id):
    """Delete network quotas for a project

        :param name_or_id: project name or id

        :returns: dict with the quotas
        :raises: :class:`~openstack.exceptions.SDKException` if it's not a
            valid project or the network client call failed
        """
    proj = self.get_project(name_or_id)
    if not proj:
        raise exceptions.SDKException('project does not exist')
    self.network.delete_quota(proj.id)