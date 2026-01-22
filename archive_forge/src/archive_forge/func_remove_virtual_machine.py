from __future__ import absolute_import, division, print_function
import re
import uuid
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import xrange
from ansible.module_utils.common.text.converters import to_native
def remove_virtual_machine(module, profitbricks):
    """
    Removes a virtual machine.

    This will remove the virtual machine along with the bootVolume.

    module : AnsibleModule object
    community.general.profitbricks: authenticated profitbricks object.

    Not yet supported: handle deletion of attached data disks.

    Returns:
        True if a new virtual server was deleted, false otherwise
    """
    datacenter = module.params.get('datacenter')
    instance_ids = module.params.get('instance_ids')
    remove_boot_volume = module.params.get('remove_boot_volume')
    changed = False
    if not isinstance(module.params.get('instance_ids'), list) or len(module.params.get('instance_ids')) < 1:
        module.fail_json(msg='instance_ids should be a list of virtual machine ids or names, aborting')
    datacenter_list = profitbricks.list_datacenters()
    datacenter_id = _get_datacenter_id(datacenter_list, datacenter)
    if not datacenter_id:
        module.fail_json(msg="Virtual data center '%s' not found." % str(datacenter))
    server_list = profitbricks.list_servers(datacenter_id)
    for instance in instance_ids:
        server_id = _get_server_id(server_list, instance)
        if server_id:
            if remove_boot_volume:
                _remove_boot_volume(module, profitbricks, datacenter_id, server_id)
            try:
                server_response = profitbricks.delete_server(datacenter_id, server_id)
            except Exception as e:
                module.fail_json(msg='failed to terminate the virtual server: %s' % to_native(e), exception=traceback.format_exc())
            else:
                changed = True
    return changed