from __future__ import (absolute_import, division, print_function)
import os
import tempfile
import copy
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
def multiple_vd_config(mod_args=None, pd_filter='', each_vd=None):
    if mod_args is None:
        mod_args = {}
    if each_vd is None:
        each_vd = {}
    if each_vd:
        mod_args.update(each_vd)
    disk_size = None
    location_list = []
    id_list = []
    size = mod_args.get('capacity')
    drives = mod_args.get('drives')
    if drives:
        if 'location' in drives:
            location_list = drives.get('location')
        elif 'id' in drives:
            id_list = drives.get('id')
    if size is not None:
        size_check = float(size)
        disk_size = '{0}'.format(int(size_check * 1073741824))
    if mod_args['media_type'] is not None:
        pd_filter += ' and disk.MediaType == "{0}"'.format(mod_args['media_type'])
    if mod_args['protocol'] is not None:
        pd_filter += ' and disk.BusProtocol == "{0}"'.format(mod_args['protocol'])
    pd_selection = pd_filter
    if location_list:
        slots = ''
        for i in location_list:
            slots += '"' + str(i) + '",'
        slots_list = '[' + slots[0:-1] + ']'
        pd_selection += ' and disk.Slot._value in ' + slots_list
    elif id_list:
        pd_selection += ' and disk.FQDD._value in ' + str(id_list)
    raid_init_operation, raid_reset_config = ('None', 'False')
    if mod_args['raid_init_operation'] == 'None':
        raid_init_operation = RAIDinitOperationTypes.T_None
    if mod_args['raid_init_operation'] == 'Fast':
        raid_init_operation = RAIDinitOperationTypes.Fast
    if mod_args['raid_reset_config'] == 'False':
        raid_reset_config = RAIDresetConfigTypes.T_False
    if mod_args['raid_reset_config'] == 'True':
        raid_reset_config = RAIDresetConfigTypes.T_True
    vd_value = dict(Name=mod_args.get('name'), SpanDepth=int(mod_args['span_depth']), SpanLength=int(mod_args['span_length']), NumberDedicatedHotSpare=int(mod_args['number_dedicated_hot_spare']), RAIDTypes=mod_args['volume_type'], DiskCachePolicy=DiskCachePolicyTypes[mod_args['disk_cache_policy']], RAIDdefaultWritePolicy=mod_args['write_cache_policy'], RAIDdefaultReadPolicy=RAIDdefaultReadPolicyTypes[mod_args['read_cache_policy']], StripeSize=int(mod_args['stripe_size']), RAIDforeignConfig='Clear', RAIDaction=RAIDactionTypes.Create, PhysicalDiskFilter=pd_selection, Size=disk_size, RAIDresetConfig=raid_reset_config, RAIDinitOperation=raid_init_operation, PDSlots=location_list, ControllerFQDD=mod_args.get('controller_id'), mediatype=mod_args['media_type'], busprotocol=mod_args['protocol'], FQDD=id_list)
    return vd_value