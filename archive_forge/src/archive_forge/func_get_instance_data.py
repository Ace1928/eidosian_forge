from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
def get_instance_data(key, cr_key, cr, existing_key):
    """
    Helper method to get instance data used to populate list structure in config
    fact dictionary
    """
    data = {}
    if existing_key is None:
        instance = None
    else:
        instance = cr._ref[cr_key]['existing'][existing_key]
    patterns = {'destination_groups': 'destination-group (\\S+)', 'sensor_groups': 'sensor-group (\\S+)', 'subscriptions': 'subscription (\\S+)'}
    if key in patterns.keys():
        m = re.search(patterns[key], cr._ref['_resource_key'])
        instance_key = m.group(1)
        data = {'id': instance_key, cr_key: instance}
    data = dict(((k, v) for k, v in data.items() if v is not None))
    return data