from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def get_balancer_state(client):
    """
    Gets the state of the MongoDB balancer. The config.settings collection does
    not exist until the balancer has been started for the first time
    { "_id" : "balancer", "mode" : "full", "stopped" : false }
    { "_id" : "autosplit", "enabled" : true }
    """
    balancer_state = None
    result = client['config'].settings.find_one({'_id': 'balancer'})
    if not result:
        balancer_state = 'stopped'
    elif result['stopped'] is False:
        balancer_state = 'started'
    else:
        balancer_state = 'stopped'
    return balancer_state