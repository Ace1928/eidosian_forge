from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def get_storages(self, type):
    """Retrieve storages information

        :param type: str, optional - type of storages
        :return: list of dicts - array of storages
        """
    try:
        return self.proxmox_api.storage.get(type=type)
    except Exception as e:
        self.module.fail_json(msg='Unable to retrieve storages information with type %s: %s' % (type, e))