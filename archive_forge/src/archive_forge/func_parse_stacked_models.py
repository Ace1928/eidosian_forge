from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.ciscosmb.plugins.module_utils.ciscosmb import (
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def parse_stacked_models(self, data):
    models = []
    for n in range(1, 9):
        if str(n) in data:
            models.append(data[str(n)]['pid'])
    return models