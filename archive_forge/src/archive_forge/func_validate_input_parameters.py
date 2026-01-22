from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
def validate_input_parameters(self, device_name=None, device_id=None, current_pathname=None, sds_name=None, sds_id=None):
    """Validate the input parameters"""
    if current_pathname:
        if (sds_name is None or len(sds_name.strip()) == 0) and (sds_id is None or len(sds_id.strip()) == 0):
            error_msg = 'sds_name or sds_id is mandatory along with current_pathname. Please enter a valid value.'
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)
    elif current_pathname is not None and len(current_pathname.strip()) == 0:
        error_msg = 'Please enter a valid value for current_pathname.'
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)
    if device_name:
        if (sds_name is None or len(sds_name.strip()) == 0) and (sds_id is None or len(sds_id.strip()) == 0):
            error_msg = 'sds_name or sds_id is mandatory along with device_name. Please enter a valid value.'
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)
    elif device_name is not None and len(device_name.strip()) == 0:
        error_msg = 'Please enter a valid value for device_name.'
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)
    if sds_name:
        if (current_pathname is None or len(current_pathname.strip()) == 0) and (device_name is None or len(device_name.strip()) == 0):
            error_msg = 'current_pathname or device_name is mandatory along with sds_name. Please enter a valid value.'
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)
    elif sds_name is not None and len(sds_name.strip()) == 0:
        error_msg = 'Please enter a valid value for sds_name.'
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)
    if sds_id:
        if (current_pathname is None or len(current_pathname.strip()) == 0) and (device_name is None or len(device_name.strip()) == 0):
            error_msg = 'current_pathname or device_name is mandatory along with sds_id. Please enter a valid value.'
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)
    elif sds_id is not None and len(sds_id.strip()) == 0:
        error_msg = 'Please enter a valid value for sds_id.'
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)
    if device_id is not None and len(device_id.strip()) == 0:
        error_msg = 'Please provide valid device_id value to identify a device.'
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)
    if current_pathname is None and device_name is None and (device_id is None):
        error_msg = 'Please specify a valid parameter combination to identify a device.'
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)