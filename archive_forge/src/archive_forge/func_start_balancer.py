from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def start_balancer(client):
    """
    Starts MongoDB balancer
    """
    cmd_doc = OrderedDict([('balancerStart', 1), ('maxTimeMS', 60000)])
    client['admin'].command(cmd_doc)
    time.sleep(1)