import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_list_projects(self):
    """
        List the available projects

        :rtype ``list`` of :class:`CloudStackProject`
        """
    res = self._sync_request(command='listProjects', method='GET')
    projs = res.get('project', [])
    projects = []
    extra_map = RESOURCE_EXTRA_ATTRIBUTES_MAP['project']
    for proj in projs:
        extra = self._get_extra_dict(proj, extra_map)
        if 'tags' in proj:
            extra['tags'] = self._get_resource_tags(proj['tags'])
        projects.append(CloudStackProject(id=proj['id'], name=proj['name'], display_text=proj['displaytext'], driver=self, extra=extra))
    return projects