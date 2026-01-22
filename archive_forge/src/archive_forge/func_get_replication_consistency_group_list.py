from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import re
def get_replication_consistency_group_list(self, filter_dict=None):
    """ Get the list of replication consistency group on a given PowerFlex storage
            system """
    try:
        LOG.info('Getting replication consistency group list ')
        if filter_dict:
            rcgs = self.powerflex_conn.replication_consistency_group.get(filter_fields=filter_dict)
        else:
            rcgs = self.powerflex_conn.replication_consistency_group.get()
        if rcgs:
            api_version = self.powerflex_conn.system.get()[0]['mdmCluster']['master']['versionInfo']
            statistics_map = self.powerflex_conn.replication_consistency_group.get_all_statistics(utils.is_version_less_than_3_6(api_version))
            list_of_rcg_ids_in_statistics = statistics_map.keys()
            for rcg in rcgs:
                rcg.pop('links', None)
                rcg['statistics'] = statistics_map[rcg['id']] if rcg['id'] in list_of_rcg_ids_in_statistics else {}
            return result_list(rcgs)
    except Exception as e:
        msg = 'Get replication consistency group list from powerflex array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)