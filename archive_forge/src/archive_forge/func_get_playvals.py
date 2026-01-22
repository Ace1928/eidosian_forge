from __future__ import absolute_import, division, print_function
import json
import re
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.module_utils.six import PY2, PY3
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
def get_playvals(self):
    """Update ref with values from the playbook.
        Store these values in each command's 'playval' key.
        """
    ref = self._ref
    module = self._module
    params = {}
    if module.params.get('config'):
        param_data = module.params.get('config')
        params['global'] = param_data
        for key in param_data.keys():
            if isinstance(param_data[key], list):
                params[key] = param_data[key]
    else:
        params['global'] = module.params
    for k in ref.keys():
        for level in params.keys():
            if isinstance(params[level], dict):
                params[level] = [params[level]]
            for item in params[level]:
                if k in item and item[k] is not None:
                    if not ref[k].get('playval'):
                        ref[k]['playval'] = {}
                    playval = item[k]
                    index = params[level].index(item)
                    if 'int' == ref[k]['kind']:
                        playval = int(playval)
                    elif 'list' == ref[k]['kind']:
                        playval = [str(i) for i in playval]
                    elif 'dict' == ref[k]['kind']:
                        for key, v in playval.items():
                            playval[key] = str(v)
                    ref[k]['playval'][index] = playval