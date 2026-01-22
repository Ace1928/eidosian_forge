from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def find_host_lun_use(module, host, volume):
    """ Return a dict showing if a host lun matches a volume. """
    check_result = {'lun_used': False, 'lun_volume_matches': False}
    desired_lun = module.params['lun']
    if desired_lun:
        for host_lun in host.get_luns():
            if desired_lun == host_lun.lun:
                if host_lun.volume == volume:
                    check_result = {'lun_used': True, 'lun_volume_matches': True}
                else:
                    check_result = {'lun_used': True, 'lun_volume_matches': False}
    return check_result