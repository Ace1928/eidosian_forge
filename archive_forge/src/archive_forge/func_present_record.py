from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ngine_io.cloudstack.plugins.module_utils.cloudstack import (
def present_record(self):
    instance = self.get_instance()
    if not instance:
        self.module.fail_json(msg='No compute instance with name=%s found. ' % self.name)
    data = {'domainname': self.content}
    record = self.get_record(instance)
    if self.has_changed(data, record):
        self.result['changed'] = True
        self.result['diff']['before'] = record
        self.result['diff']['after'] = data
        if not self.module.check_mode:
            self.query_api('updateReverseDnsForVirtualMachine', id=instance['id'], domainname=data['domainname'])
    return data