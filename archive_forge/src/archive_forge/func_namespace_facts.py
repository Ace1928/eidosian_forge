from __future__ import (absolute_import, division, print_function)
import os
import re
from collections.abc import MutableMapping, MutableSequence
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils import six
from ansible.plugins.loader import connection_loader
from ansible.utils.display import Display
def namespace_facts(facts):
    """ return all facts inside 'ansible_facts' w/o an ansible_ prefix """
    deprefixed = {}
    for k in facts:
        if k.startswith('ansible_') and k not in ('ansible_local',):
            deprefixed[k[8:]] = module_response_deepcopy(facts[k])
        else:
            deprefixed[k] = module_response_deepcopy(facts[k])
    return {'ansible_facts': deprefixed}