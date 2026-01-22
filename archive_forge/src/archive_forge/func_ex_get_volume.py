import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_get_volume(self, volume_id, project=None):
    """
        Return a StorageVolume object based on its ID.

        :param  volume_id: The id of the volume
        :type   volume_id: ``str``

        :keyword    project: Limit volume returned to those configured under
                             the defined project.
        :type       project: :class:`.CloudStackProject`

        :rtype: :class:`CloudStackNode`
        """
    args = {'id': volume_id}
    if project:
        args['projectid'] = project.id
    volumes = self._sync_request(command='listVolumes', params=args)
    if not volumes:
        raise Exception("Volume '%s' not found" % volume_id)
    vol = volumes['volume'][0]
    extra_map = RESOURCE_EXTRA_ATTRIBUTES_MAP['volume']
    extra = self._get_extra_dict(vol, extra_map)
    if 'tags' in vol:
        extra['tags'] = self._get_resource_tags(vol['tags'])
    state = self._to_volume_state(vol)
    volume = StorageVolume(id=vol['id'], name=vol['name'], state=state, size=vol['size'], driver=self, extra=extra)
    return volume