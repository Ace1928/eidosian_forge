from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
import ipaddress
from ipaddress import ip_network
def get_nas_server_obj(self, nas_server_name, nas_server_id):
    """Get NAS server ID.
            :param: nas_server_name: The name of NAS server
            :param: nas_server_id: ID of NAS server
            :return: Return NAS server object if exists
        """
    LOG.info('Getting NAS server object')
    try:
        if nas_server_name:
            obj_nas = self.unity_conn.get_nas_server(name=nas_server_name)
            return obj_nas
        elif nas_server_id:
            obj_nas = self.unity_conn.get_nas_server(_id=nas_server_id)
            if obj_nas._get_properties()['existed']:
                return obj_nas
            else:
                msg = 'NAS server with id %s does not exist' % nas_server_id
                LOG.error(msg)
                self.module.fail_json(msg=msg)
    except Exception as e:
        msg = 'Failed to get details of NAS server with error: %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)