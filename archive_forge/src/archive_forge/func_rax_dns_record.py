from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import (rax_argument_spec,
def rax_dns_record(module, comment=None, data=None, domain=None, name=None, overwrite=True, priority=None, record_type='A', state='present', ttl=7200):
    """Function for manipulating record types other than PTR"""
    changed = False
    dns = pyrax.cloud_dns
    if not dns:
        module.fail_json(msg='Failed to instantiate client. This typically indicates an invalid region or an incorrectly capitalized region name.')
    if state == 'present':
        if not priority and record_type in ['MX', 'SRV']:
            module.fail_json(msg='A "priority" attribute is required for creating a MX or SRV record')
        try:
            domain = dns.find(name=domain)
        except Exception as e:
            module.fail_json(msg='%s' % e.message)
        try:
            if overwrite:
                record = domain.find_record(record_type, name=name)
            else:
                record = domain.find_record(record_type, name=name, data=data)
        except pyrax.exceptions.DomainRecordNotUnique as e:
            module.fail_json(msg='overwrite=true and there are multiple matching records')
        except pyrax.exceptions.DomainRecordNotFound as e:
            try:
                record_data = {'type': record_type, 'name': name, 'data': data, 'ttl': ttl}
                if comment:
                    record_data.update(dict(comment=comment))
                if priority and record_type.upper() in ['MX', 'SRV']:
                    record_data.update(dict(priority=priority))
                record = domain.add_records([record_data])[0]
                changed = True
            except Exception as e:
                module.fail_json(msg='%s' % e.message)
        update = {}
        if comment != getattr(record, 'comment', None):
            update['comment'] = comment
        if ttl != getattr(record, 'ttl', None):
            update['ttl'] = ttl
        if priority != getattr(record, 'priority', None):
            update['priority'] = priority
        if data != getattr(record, 'data', None):
            update['data'] = data
        if update:
            try:
                record.update(**update)
                changed = True
                record.get()
            except Exception as e:
                module.fail_json(msg='%s' % e.message)
    elif state == 'absent':
        try:
            domain = dns.find(name=domain)
        except Exception as e:
            module.fail_json(msg='%s' % e.message)
        try:
            record = domain.find_record(record_type, name=name, data=data)
        except pyrax.exceptions.DomainRecordNotFound as e:
            record = {}
        except pyrax.exceptions.DomainRecordNotUnique as e:
            module.fail_json(msg='%s' % e.message)
        if record:
            try:
                record.delete()
                changed = True
            except Exception as e:
                module.fail_json(msg='%s' % e.message)
    module.exit_json(changed=changed, record=rax_to_dict(record))