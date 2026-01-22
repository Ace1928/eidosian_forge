from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def update_pod(module, array):
    """Update Pod configuration"""
    changed = False
    current_config = array.get_pod(module.params['name'], failover_preference=True)
    if module.params['failover']:
        current_failover = current_config['failover_preference']
        if current_failover == [] or sorted(module.params['failover']) != sorted(current_failover):
            changed = True
            if not module.check_mode:
                try:
                    if module.params['failover'] == ['auto']:
                        if current_failover != []:
                            array.set_pod(module.params['name'], failover_preference=[])
                    else:
                        array.set_pod(module.params['name'], failover_preference=module.params['failover'])
                except Exception:
                    module.fail_json(msg='Failed to set failover preference for pod {0}.'.format(module.params['name']))
    current_config = array.get_pod(module.params['name'], mediator=True)
    if current_config['mediator'] != module.params['mediator']:
        changed = True
        if not module.check_mode:
            try:
                array.set_pod(module.params['name'], mediator=module.params['mediator'])
            except Exception:
                module.warn('Failed to communicate with mediator {0}. Setting unchanged.'.format(module.params['mediator']))
    if module.params['promote'] is not None:
        if len(current_config['arrays']) > 1:
            module.fail_json(msg='Promotion/Demotion not permitted. Pod {0} is stretched'.format(module.params['name']))
        elif current_config['promotion_status'] == 'demoted' and module.params['promote']:
            try:
                if module.params['undo'] is None:
                    module.params['undo'] = True
                if current_config['promotion_status'] == 'quiescing':
                    module.fail_json(msg='Cannot promote pod {0} as it is still quiesing'.format(module.params['name']))
                elif module.params['undo']:
                    changed = True
                    if not module.check_mode:
                        if get_undo_pod(module, array):
                            array.promote_pod(module.params['name'], promote_from=module.params['name'] + '.undo-demote')
                        else:
                            array.promote_pod(module.params['name'])
                            module.warn('undo-demote pod remaining for {0}. Consider eradicating this.'.format(module.params['name']))
                else:
                    changed = True
                    if not module.check_mode:
                        array.promote_pod(module.params['name'])
            except Exception:
                module.fail_json(msg='Failed to promote pod {0}.'.format(module.params['name']))
        elif current_config['promotion_status'] != 'demoted' and (not module.params['promote']):
            try:
                if get_undo_pod(module, array):
                    module.fail_json(msg='Cannot demote pod {0} due to associated undo-demote pod not being eradicated'.format(module.params['name']))
                if module.params['quiesce'] is None:
                    module.params['quiesce'] = True
                if current_config['link_target_count'] == 0:
                    changed = True
                    if not module.check_mode:
                        array.demote_pod(module.params['name'])
                elif not module.params['quiesce']:
                    changed = True
                    if not module.check_mode:
                        array.demote_pod(module.params['name'], skip_quiesce=True)
                else:
                    changed = True
                    if not module.check_mode:
                        array.demote_pod(module.params['name'], quiesce=True)
            except Exception:
                module.fail_json(msg='Failed to demote pod {0}.'.format(module.params['name']))
    if module.params['quota']:
        arrayv6 = get_array(module)
        current_pod = list(arrayv6.get_pods(names=[module.params['name']]).items)[0]
        quota = human_to_bytes(module.params['quota'])
        if current_pod.quota_limit != quota:
            changed = True
            if not module.check_mode:
                res = arrayv6.patch_pods(names=[module.params['name']], pod=flasharray.PodPatch(quota_limit=quota, ignore_usage=module.params['ignore_usage']))
                if res.status_code != 200:
                    module.fail_json(msg='Failed to update quota on pod {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)