from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import rax_argument_spec, rax_required_together, setup_rax_module
def cloud_check(module, state, entity_id, label, check_type, monitoring_zones_poll, target_hostname, target_alias, details, disabled, metadata, period, timeout):
    if monitoring_zones_poll and (not isinstance(monitoring_zones_poll, list)):
        monitoring_zones_poll = [monitoring_zones_poll]
    if period:
        period = int(period)
    if timeout:
        timeout = int(timeout)
    changed = False
    check = None
    cm = pyrax.cloud_monitoring
    if not cm:
        module.fail_json(msg='Failed to instantiate client. This typically indicates an invalid region or an incorrectly capitalized region name.')
    entity = cm.get_entity(entity_id)
    if not entity:
        module.fail_json(msg='Failed to instantiate entity. "%s" may not be a valid entity id.' % entity_id)
    existing = [e for e in entity.list_checks() if e.label == label]
    if existing:
        check = existing[0]
    if state == 'present':
        if len(existing) > 1:
            module.fail_json(msg='%s existing checks have a label of %s.' % (len(existing), label))
        should_delete = False
        should_create = False
        should_update = False
        if check:
            if details:
                for key, value in details.items():
                    if key not in check.details:
                        should_delete = should_create = True
                    elif value != check.details[key]:
                        should_delete = should_create = True
            should_update = label != check.label or (target_hostname and target_hostname != check.target_hostname) or (target_alias and target_alias != check.target_alias) or (disabled != check.disabled) or (metadata and metadata != check.metadata) or (period and period != check.period) or (timeout and timeout != check.timeout) or (monitoring_zones_poll and monitoring_zones_poll != check.monitoring_zones_poll)
            if should_update and (not should_delete):
                check.update(label=label, disabled=disabled, metadata=metadata, monitoring_zones_poll=monitoring_zones_poll, timeout=timeout, period=period, target_alias=target_alias, target_hostname=target_hostname)
                changed = True
        else:
            should_create = True
        if should_delete:
            check.delete()
        if should_create:
            check = cm.create_check(entity, label=label, check_type=check_type, target_hostname=target_hostname, target_alias=target_alias, monitoring_zones_poll=monitoring_zones_poll, details=details, disabled=disabled, metadata=metadata, period=period, timeout=timeout)
            changed = True
    elif state == 'absent':
        if check:
            check.delete()
            changed = True
    else:
        module.fail_json(msg='state must be either present or absent.')
    if check:
        check_dict = {'id': check.id, 'label': check.label, 'type': check.type, 'target_hostname': check.target_hostname, 'target_alias': check.target_alias, 'monitoring_zones_poll': check.monitoring_zones_poll, 'details': check.details, 'disabled': check.disabled, 'metadata': check.metadata, 'period': check.period, 'timeout': check.timeout}
        module.exit_json(changed=changed, check=check_dict)
    else:
        module.exit_json(changed=changed)