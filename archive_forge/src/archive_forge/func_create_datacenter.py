from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
def create_datacenter(module, profitbricks):
    """
    Creates a Datacenter

    This will create a new Datacenter in the specified location.

    module : AnsibleModule object
    profitbricks: authenticated profitbricks object.

    Returns:
        True if a new datacenter was created, false otherwise
    """
    name = module.params.get('name')
    location = module.params.get('location')
    description = module.params.get('description')
    wait = module.params.get('wait')
    wait_timeout = int(module.params.get('wait_timeout'))
    i = Datacenter(name=name, location=location, description=description)
    try:
        datacenter_response = profitbricks.create_datacenter(datacenter=i)
        if wait:
            _wait_for_completion(profitbricks, datacenter_response, wait_timeout, '_create_datacenter')
        results = {'datacenter_id': datacenter_response['id']}
        return results
    except Exception as e:
        module.fail_json(msg='failed to create the new datacenter: %s' % str(e))