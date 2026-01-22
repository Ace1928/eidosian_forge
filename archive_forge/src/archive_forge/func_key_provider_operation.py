from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def key_provider_operation(self):
    results = {'failed': False, 'changed': False}
    kp_name = self.params['name']
    if not kp_name:
        self.module.fail_json(msg="Please set a valid name of key provider via 'name' parameter, now it's '%s'," % kp_name)
    key_provider_clusters = self.get_key_provider_clusters()
    existing_kp_cluster = self.get_key_provider_by_name(key_provider_clusters, kp_name)
    existing_kp_type = self.get_key_provider_type(existing_kp_cluster)
    if existing_kp_cluster is not None:
        if existing_kp_type and existing_kp_type == 'native':
            self.module.fail_json(msg="Native Key Provider with name '%s' already exist, please change to another name for Standard Key Provider operation using this module." % kp_name)
        self.key_provider_id = existing_kp_cluster.clusterId
    if self.params['state'] == 'present':
        is_default_kp = False
        proxy_user_config = dict()
        proxy_user_config.update(proxy_server=self.params.get('proxy_server'), proxy_port=self.params.get('proxy_port'), kms_username=self.params.get('kms_username'), kms_password=self.params.get('kms_password'))
        if existing_kp_cluster is not None:
            is_default_kp = existing_kp_cluster.useAsDefault
            if self.module.check_mode:
                results['desired_operation'] = 'reconfig standard key provider'
                results['target_key_provider'] = kp_name
                self.module.exit_json(**results)
            else:
                results['operation'] = 'reconfig standard key provider'
                results['changed'] = self.reconfig_kmip_standard_kp(existing_kp_cluster.servers, self.params['kms_info'], proxy_user_config)
        else:
            if len(self.params['kms_info']) == 0:
                self.module.fail_json(msg="Please set 'kms_info' when add new standard key provider")
            for configured_kms_info in self.params['kms_info']:
                if configured_kms_info.get('remove_kms'):
                    self.module.fail_json(msg="Specified key provider '%s' not exist, so no KMS server to be removed." % kp_name)
            if self.module.check_mode:
                results['desired_operation'] = 'add standard key provider'
                self.module.exit_json(**results)
            else:
                results['operation'] = 'add standard key provider'
                new_key_provider_id = self.setup_standard_kp(kp_name, self.params['kms_info'], proxy_user_config)
                if new_key_provider_id:
                    self.key_provider_id = new_key_provider_id
                    if len(key_provider_clusters) == 0:
                        self.params['mark_default'] = True
                    results['changed'] = True
        if self.key_provider_id and self.params['mark_default'] and (not is_default_kp):
            self.set_default_key_provider()
            results['changed'] = True
        if self.key_provider_id and self.params.get('make_kms_trust_vc'):
            results['changed'], cert_info = self.download_upload_cert_for_trust(self.params['make_kms_trust_vc'])
            results['msg'] = cert_info
    else:
        if self.module.check_mode:
            results['desired_operation'] = 'remove standard key provider'
        else:
            results['operation'] = 'remove standard key provider'
        if existing_kp_cluster is None:
            output_msg = "Key Provider with name '%s' is not found." % kp_name
            if self.module.check_mode:
                results['msg'] = output_msg
                self.module.exit_json(**results)
            else:
                self.module.fail_json(msg=output_msg)
        elif self.module.check_mode:
            results['target_key_provider'] = kp_name
            self.module.exit_json(**results)
        else:
            self.remove_kms_cluster(existing_kp_cluster)
            results['changed'] = True
    if results['changed']:
        key_provider_clusters = self.get_key_provider_clusters()
    results['key_provider_clusters'] = self.gather_key_provider_cluster_info(key_provider_clusters)
    self.module.exit_json(**results)