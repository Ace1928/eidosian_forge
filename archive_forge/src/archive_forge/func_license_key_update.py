from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def license_key_update(self):
    existing_license_keys = []
    license_id_pairs = {}
    license_add_remove = False
    if self.module.check_mode:
        self.changed = True
        return
    for key in self.license_key:
        if key == 'None':
            self.log(' Empty License key list provided')
            return
    cmd = 'lsfeature'
    cmdopts = {}
    feature_list = self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
    for feature in feature_list:
        existing_license_keys.append(feature['license_key'])
        license_id_pairs[feature['license_key']] = feature['id']
    self.log('existing licenses=%s, license_id_pairs=%s', existing_license_keys, license_id_pairs)
    if set(existing_license_keys).symmetric_difference(set(self.license_key)):
        license_add_remove = True
    if license_add_remove:
        deactivate_license_keys, activate_license_keys = (False, False)
        deactivate_license_keys = list(set(existing_license_keys) - set(self.license_key))
        self.log('deactivate_license_keys %s ', deactivate_license_keys)
        if deactivate_license_keys:
            for item in deactivate_license_keys:
                if not item:
                    self.log('%s item', [license_id_pairs[item]])
                    self.restapi.svc_run_command('deactivatefeature', None, [license_id_pairs[item]])
                    self.changed = True
                    self.log('%s deactivated', deactivate_license_keys)
            self.message += ' License %s deactivated.' % deactivate_license_keys
        activate_license_keys = list(set(self.license_key) - set(existing_license_keys))
        self.log('activate_license_keys %s ', activate_license_keys)
        if activate_license_keys:
            for item in activate_license_keys:
                if item:
                    self.restapi.svc_run_command('activatefeature', {'licensekey': item}, None)
                    self.changed = True
                    self.log('%s activated', activate_license_keys)
            self.message += ' License %s activated.' % activate_license_keys
    else:
        self.message += ' No license Changes.'