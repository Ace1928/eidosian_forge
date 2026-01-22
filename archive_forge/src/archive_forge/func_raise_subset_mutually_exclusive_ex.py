from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
import re
def raise_subset_mutually_exclusive_ex():
    msg = "Subset 'all' and 'minimum' are mutually exclusive. Please provide only one of them"
    raise Exception(msg)