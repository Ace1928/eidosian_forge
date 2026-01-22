from cinderclient.apiclient import base as common_base
from cinderclient import base
def set_bootable(self, volume, flag):
    return self._action('os-set_bootable', base.getid(volume), {'bootable': flag})