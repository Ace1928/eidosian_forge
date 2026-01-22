from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import rax_argument_spec, rax_required_together, setup_rax_module
def cloud_monitoring(module, state, label, agent_id, named_ip_addresses, metadata):
    if len(label) < 1 or len(label) > 255:
        module.fail_json(msg='label must be between 1 and 255 characters long')
    changed = False
    cm = pyrax.cloud_monitoring
    if not cm:
        module.fail_json(msg='Failed to instantiate client. This typically indicates an invalid region or an incorrectly capitalized region name.')
    existing = []
    for entity in cm.list_entities():
        if label == entity.label:
            existing.append(entity)
    entity = None
    if existing:
        entity = existing[0]
    if state == 'present':
        should_update = False
        should_delete = False
        should_create = False
        if len(existing) > 1:
            module.fail_json(msg='%s existing entities have the label %s.' % (len(existing), label))
        if entity:
            if named_ip_addresses and named_ip_addresses != entity.ip_addresses:
                should_delete = should_create = True
            should_update = agent_id and agent_id != entity.agent_id or (metadata and metadata != entity.metadata)
            if should_update and (not should_delete):
                entity.update(agent_id, metadata)
                changed = True
            if should_delete:
                entity.delete()
        else:
            should_create = True
        if should_create:
            entity = cm.create_entity(label=label, agent=agent_id, ip_addresses=named_ip_addresses, metadata=metadata)
            changed = True
    else:
        for e in existing:
            e.delete()
            changed = True
    if entity:
        entity_dict = {'id': entity.id, 'name': entity.name, 'agent_id': entity.agent_id}
        module.exit_json(changed=changed, entity=entity_dict)
    else:
        module.exit_json(changed=changed)