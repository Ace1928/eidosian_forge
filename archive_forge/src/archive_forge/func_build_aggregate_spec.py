from __future__ import absolute_import, division, print_function
import json
import re
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, ComplexList
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
def build_aggregate_spec(element_spec, required, *extra_spec):
    aggregate_spec = deepcopy(element_spec)
    for elt in required:
        aggregate_spec[elt] = dict(required=True)
    remove_default_spec(aggregate_spec)
    argument_spec = dict(aggregate=dict(type='list', elements='dict', options=aggregate_spec))
    argument_spec.update(element_spec)
    for elt in extra_spec:
        argument_spec.update(elt)
    return argument_spec