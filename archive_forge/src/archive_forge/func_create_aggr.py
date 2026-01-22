from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def create_aggr(self):
    """
        Create aggregate
        :return: None
        """
    if self.use_rest:
        return self.create_aggr_rest()
    options = {'aggregate': self.parameters['name']}
    if self.parameters.get('disk_class'):
        options['disk-class'] = self.parameters['disk_class']
    if self.parameters.get('disk_type'):
        options['disk-type'] = self.parameters['disk_type']
    if self.parameters.get('raid_type'):
        options['raid-type'] = self.parameters['raid_type']
    if self.parameters.get('snaplock_type'):
        options['snaplock-type'] = self.parameters['snaplock_type']
    if self.parameters.get('spare_pool'):
        options['spare-pool'] = self.parameters['spare_pool']
    if self.parameters.get('disk_count'):
        options['disk-count'] = str(self.parameters['disk_count'])
    if self.parameters.get('disk_size'):
        options['disk-size'] = str(self.parameters['disk_size'])
    if self.parameters.get('disk_size_with_unit'):
        options['disk-size-with-unit'] = str(self.parameters['disk_size_with_unit'])
    if self.parameters.get('raid_size'):
        options['raid-size'] = str(self.parameters['raid_size'])
    if self.parameters.get('is_mirrored'):
        options['is-mirrored'] = str(self.parameters['is_mirrored']).lower()
    if self.parameters.get('ignore_pool_checks'):
        options['ignore-pool-checks'] = str(self.parameters['ignore_pool_checks']).lower()
    if self.parameters.get('encryption'):
        options['encrypt-with-aggr-key'] = str(self.parameters['encryption']).lower()
    aggr_create = netapp_utils.zapi.NaElement.create_node_with_children('aggr-create', **options)
    if self.parameters.get('nodes'):
        nodes_obj = netapp_utils.zapi.NaElement('nodes')
        aggr_create.add_child_elem(nodes_obj)
        for node in self.parameters['nodes']:
            nodes_obj.add_new_child('node-name', node)
    if self.parameters.get('disks'):
        aggr_create.add_child_elem(self.get_disks_or_mirror_disks_object('disks', self.parameters.get('disks')))
    if self.parameters.get('mirror_disks'):
        aggr_create.add_child_elem(self.get_disks_or_mirror_disks_object('mirror-disks', self.parameters.get('mirror_disks')))
    try:
        self.server.invoke_successfully(aggr_create, enable_tunneling=False)
        if self.parameters.get('wait_for_online'):
            retries = (self.parameters['time_out'] + 5) / 10
            current = self.get_aggr()
            status = None if current is None else current['service_state']
            while status != 'online' and retries > 0:
                time.sleep(10)
                retries = retries - 1
                current = self.get_aggr()
                status = None if current is None else current['service_state']
        else:
            current = self.get_aggr()
        if current is not None and current.get('disk_count') != self.parameters.get('disk_count'):
            self.module.warn('Aggregate created with mismatched disk_count: created %s not %s' % (current.get('disk_count'), self.parameters.get('disk_count')))
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error provisioning aggregate %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())