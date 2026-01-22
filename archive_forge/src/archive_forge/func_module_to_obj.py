from __future__ import absolute_import, division, print_function
import copy
import json
from ansible.module_utils.six.moves.urllib import error as urllib_error
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import open_url
def module_to_obj(self, is_update):
    obj = {}
    for k, v in self.params.items():
        result = self.map_param(k, v, is_update)
        if result:
            obj[result[0]] = result[1]
    return obj