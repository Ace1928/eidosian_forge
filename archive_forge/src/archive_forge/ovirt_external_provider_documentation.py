from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (

        Update auth keys for volume provider, if not exist add them or remove
        if they are not specified and there are already defined in the external
        volume provider.

        Args:
            provider (dict): Volume provider details.
            providers_service (openstack_volume_providers_service): Provider
                service.
            keys (list): Keys to be updated/added to volume provider, each key
                is represented as dict with keys: uuid, value.
        