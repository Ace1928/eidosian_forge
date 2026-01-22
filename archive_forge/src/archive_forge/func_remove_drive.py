from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
def remove_drive(self, drives=None):
    """
        Remove Drive active in Cluster
        """
    kwargs = dict()
    if self.force_during_upgrade is not None:
        kwargs['force_during_upgrade'] = self.force_during_upgrade
    try:
        self.sfe.remove_drives(drives, **kwargs)
    except Exception as exception_object:
        self.module.fail_json(msg='Error removing drive%s: %s: %s' % ('s' if len(drives) > 1 else '', str(drives), to_native(exception_object)), exception=traceback.format_exc())