from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_volume_display_attributes(self, obj_vol):
    """get display volume attributes
        :param obj_vol: volume instance
        :return: volume dict to display
        """
    try:
        obj_vol = obj_vol.update()
        volume_details = obj_vol._get_properties()
        volume_details['size_total_with_unit'] = utils.convert_size_with_unit(int(volume_details['size_total']))
        volume_details.update({'host_access': self.get_volume_host_access_list(obj_vol)})
        if obj_vol.snap_schedule:
            volume_details.update({'snap_schedule': {'name': obj_vol.snap_schedule.name, 'id': obj_vol.snap_schedule.id}})
        if obj_vol.io_limit_policy:
            volume_details.update({'io_limit_policy': {'name': obj_vol.io_limit_policy.id, 'id': obj_vol.io_limit_policy.id}})
        if obj_vol.pool:
            volume_details.update({'pool': {'name': obj_vol.pool.name, 'id': obj_vol.pool.id}})
        return volume_details
    except Exception as e:
        errormsg = 'Failed to display the volume {0} with error {1}'.format(obj_vol.name, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)