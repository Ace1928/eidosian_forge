from __future__ import absolute_import, division, print_function
import os
import re
import sys
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.respawn import has_respawned, respawn_module
from ansible.module_utils.common.text.converters import to_native
def query_atom(module, atom, action):
    vdb = vartree.vardbapi()
    try:
        exists = vdb.match(atom)
    except InvalidAtom:
        return False
    return bool(exists)