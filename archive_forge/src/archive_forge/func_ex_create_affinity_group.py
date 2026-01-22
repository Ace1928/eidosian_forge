import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_create_affinity_group(self, name, group_type):
    """
        Creates a new Affinity Group

        :param name: Name of the affinity group
        :type  name: ``str``

        :param group_type: Type of the affinity group from the available
                           affinity/anti-affinity group types
        :type  group_type: :class:`CloudStackAffinityGroupType`

        :param description: Optional description of the affinity group
        :type  description: ``str``

        :param domainid: domain ID of the account owning the affinity group
        :type  domainid: ``str``

        :rtype: :class:`CloudStackAffinityGroup`
        """
    for ag in self.ex_list_affinity_groups():
        if name == ag.name:
            raise LibcloudError('This Affinity Group name already exists')
    params = {'name': name, 'type': group_type.type}
    result = self._async_request(command='createAffinityGroup', params=params, method='GET')
    return self._to_affinity_group(result['affinitygroup'])