from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_volume_host_access_list(self, obj_vol):
    """
        Get volume host access list
        :param obj_vol: volume instance
        :return: host list
        """
    host_list = []
    if obj_vol.host_access:
        for host_access in obj_vol.host_access:
            host = self.get_host(host_id=host_access.host.id).update()
            hlu = None
            for host_lun in host.host_luns:
                if host_lun.lun.name == obj_vol.name:
                    hlu = host_lun.hlu
            host_list.append({'name': host_access.host.name, 'id': host_access.host.id, 'accessMask': host_access.access_mask.name, 'hlu': hlu})
    return host_list