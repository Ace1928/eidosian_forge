from __future__ import absolute_import, division, print_function
import json
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six import PY3
from ansible.module_utils.common.text.converters import to_native
def create_annotation(module):
    """ Takes ansible module object """
    annotation = {}
    duration = module.params['duration']
    if module.params['start'] is not None:
        start = module.params['start']
    else:
        start = int(time.time())
    if module.params['stop'] is not None:
        stop = module.params['stop']
    else:
        stop = int(time.time()) + duration
    annotation['start'] = start
    annotation['stop'] = stop
    annotation['category'] = module.params['category']
    annotation['description'] = module.params['description']
    annotation['title'] = module.params['title']
    return annotation