from __future__ import (absolute_import, division, print_function)
import json
import re
import time
import os
from ansible.plugins.inventory import BaseInventoryPlugin
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible.module_utils.six import raise_from
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.community.general.plugins.module_utils.lxd import LXDClient, LXDClientException
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def get_network_data(self, names):
    """Create Inventory of the instance

        Iterate through the different branches of the instances and collect Information.

        Args:
            list(names): List of instance names
        Kwargs:
            None
        Raises:
            None
        Returns:
            None"""
    branches = [('networks', 'state')]
    network_config = {}
    for branch in branches:
        for name in names:
            try:
                network_config['networks'] = self._get_config(branch, name)
            except LXDClientException:
                network_config['networks'] = {name: None}
            self.data = dict_merge(network_config, self.data)