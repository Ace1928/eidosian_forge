from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@api_wrapper
def create_filesystem(module, system):
    """ Create Filesystem """
    changed = False
    if not module.check_mode:
        if module.params['thin_provision']:
            provisioning = 'THIN'
        else:
            provisioning = 'THICK'
        filesystem = system.filesystems.create(name=module.params['name'], provtype=provisioning, pool=get_pool(module, system))
        if module.params['size']:
            size = Capacity(module.params['size']).roundup(64 * KiB)
            filesystem.update_size(size)
        is_write_prot = filesystem.is_write_protected()
        desired_is_write_prot = module.params['write_protected']
        if is_write_prot != desired_is_write_prot:
            filesystem.update_field('write_protected', desired_is_write_prot)
        changed = True
    return changed