from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def update_hostgroup(module, array):
    changed = False
    renamed = False
    hgroup = get_hostgroup(module, array)
    current_hostgroup = module.params['name']
    volumes = array.list_hgroup_connections(module.params['name'])
    if module.params['state'] == 'present':
        if module.params['rename']:
            if not rename_exists(module, array):
                try:
                    if not module.check_mode:
                        array.rename_hgroup(module.params['name'], module.params['rename'])
                    current_hostgroup = module.params['rename']
                    renamed = True
                except Exception:
                    module.fail_json(msg='Rename to {0} failed.'.format(module.params['rename']))
            else:
                module.warn('Rename failed. Hostgroup {0} already exists. Continuing with other changes...'.format(module.params['rename']))
        if module.params['host']:
            cased_hosts = list(module.params['host'])
            cased_hghosts = list(hgroup['hosts'])
            new_hosts = list(set(cased_hosts).difference(cased_hghosts))
            if new_hosts:
                try:
                    if not module.check_mode:
                        array.set_hgroup(current_hostgroup, addhostlist=new_hosts)
                    changed = True
                except Exception:
                    module.fail_json(msg='Failed to add host(s) to hostgroup')
        if module.params['volume']:
            if volumes:
                current_vols = [vol['vol'] for vol in volumes]
                cased_vols = list(module.params['volume'])
                new_volumes = list(set(cased_vols).difference(set(current_vols)))
                if len(new_volumes) == 1 and module.params['lun']:
                    try:
                        if not module.check_mode:
                            array.connect_hgroup(current_hostgroup, new_volumes[0], lun=module.params['lun'])
                        changed = True
                    except Exception:
                        module.fail_json(msg='Failed to add volume {0} with LUN ID {1}'.format(new_volumes[0], module.params['lun']))
                else:
                    for cvol in new_volumes:
                        try:
                            if not module.check_mode:
                                array.connect_hgroup(current_hostgroup, cvol)
                            changed = True
                        except Exception:
                            module.fail_json(msg='Failed to connect volume {0} to hostgroup {1}.'.format(cvol, current_hostgroup))
            elif len(module.params['volume']) == 1 and module.params['lun']:
                try:
                    if not module.check_mode:
                        array.connect_hgroup(current_hostgroup, module.params['volume'][0], lun=module.params['lun'])
                    changed = True
                except Exception:
                    module.fail_json(msg='Failed to add volume {0} with LUN ID {1}'.format(module.params['volume'], module.params['lun']))
            else:
                for cvol in module.params['volume']:
                    try:
                        if not module.check_mode:
                            array.connect_hgroup(current_hostgroup, cvol)
                        changed = True
                    except Exception:
                        module.fail_json(msg='Failed to connect volume {0} to hostgroup {1}.'.format(cvol, current_hostgroup))
    else:
        if module.params['host']:
            cased_old_hosts = list(module.params['host'])
            cased_hosts = list(hgroup['hosts'])
            old_hosts = list(set(cased_old_hosts).intersection(cased_hosts))
            if old_hosts:
                try:
                    if not module.check_mode:
                        array.set_hgroup(current_hostgroup, remhostlist=old_hosts)
                    changed = True
                except Exception:
                    module.fail_json(msg='Failed to remove hosts {0} from hostgroup {1}'.format(old_hosts, current_hostgroup))
        if module.params['volume']:
            cased_old_vols = list(module.params['volume'])
            old_volumes = list(set(cased_old_vols).intersection(set([vol['vol'] for vol in volumes])))
            if old_volumes:
                changed = True
                for cvol in old_volumes:
                    try:
                        if not module.check_mode:
                            array.disconnect_hgroup(current_hostgroup, cvol)
                    except Exception:
                        module.fail_json(msg='Failed to disconnect volume {0} from hostgroup {1}'.format(cvol, current_hostgroup))
    changed = changed or renamed
    module.exit_json(changed=changed)