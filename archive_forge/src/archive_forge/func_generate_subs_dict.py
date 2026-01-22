from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
import time
def generate_subs_dict(array):
    subs_info = {}
    subs = list(array.get_subscription_assets().items)
    for sub in range(0, len(subs)):
        name = subs[sub].name
        subs_info[name] = {'subscription_id': subs[sub].subscription.id}
    return subs_info