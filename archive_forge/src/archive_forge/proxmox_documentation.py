from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
Retrieve storages information

        :param type: str, optional - type of storages
        :return: list of dicts - array of storages
        