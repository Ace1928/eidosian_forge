from __future__ import absolute_import, division, print_function
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def validate_volumes(self, volumes):
    """Validate the volumes.
            :param volumes: List of volumes
        """
    for vol in volumes:
        if 'vol_id' in vol and 'vol_name' in vol:
            errormsg = 'Both name and id are found for volume {0}. No action would be taken. Please specify either name or id.'.format(vol)
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
        elif 'vol_id' in vol and len(vol['vol_id'].strip()) == 0:
            errormsg = 'vol_id is blank. Please specify valid vol_id.'
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
        elif 'vol_name' in vol and len(vol.get('vol_name').strip()) == 0:
            errormsg = 'vol_name is blank. Please specify valid vol_name.'
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
        elif 'vol_name' in vol:
            self.get_volume_details(vol_name=vol['vol_name'])
        elif 'vol_id' in vol:
            self.get_volume_details(vol_id=vol['vol_id'])
        else:
            errormsg = 'Expected either vol_name or vol_id, found neither for volume {0}'.format(vol)
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)