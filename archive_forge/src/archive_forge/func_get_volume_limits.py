import warnings
from openstack.block_storage.v3._proxy import Proxy
from openstack.block_storage.v3 import quota_set as _qs
from openstack.cloud import _utils
from openstack import exceptions
from openstack import warnings as os_warnings
def get_volume_limits(self, name_or_id=None):
    """Get volume limits for the current project

        :param name_or_id: (optional) Project name or ID to get limits for
            if different from the current project
        :returns: The volume ``Limit`` object if found, else None.
        """
    params = {}
    if name_or_id:
        project = self.get_project(name_or_id)
        if not project:
            raise exceptions.SDKException('project does not exist')
        params['project'] = project
    return self.block_storage.get_limits(**params)