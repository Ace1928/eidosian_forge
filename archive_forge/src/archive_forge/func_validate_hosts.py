from __future__ import absolute_import, division, print_function
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def validate_hosts(self, hosts):
    """Validate hosts.
            :param hosts: List of hosts
        """
    for host in hosts:
        if 'host_id' in host and 'host_name' in host:
            errormsg = 'Both name and id are found for host {0}. No action would be taken. Please specify either name or id.'.format(host)
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
        elif 'host_id' in host and len(host['host_id'].strip()) == 0:
            errormsg = 'host_id is blank. Please specify valid host_id.'
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
        elif 'host_name' in host and len(host.get('host_name').strip()) == 0:
            errormsg = 'host_name is blank. Please specify valid host_name.'
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
        elif 'host_name' in host:
            self.get_host_id_by_name(host_name=host['host_name'])
        elif 'host_id' in host:
            host_obj = self.unity_conn.get_host(_id=host['host_id'])
            if host_obj is None or host_obj.existed is False:
                msg = 'Host id: %s does not exists' % host['host_id']
                LOG.error(msg)
                self.module.fail_json(msg=msg)
        else:
            errormsg = 'Expected either host_name or host_id, found neither for host {0}'.format(host)
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)