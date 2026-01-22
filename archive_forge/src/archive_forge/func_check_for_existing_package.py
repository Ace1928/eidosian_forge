from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def check_for_existing_package(self, error):
    """ ONTAP returns 'Package image with the same name already exists'
            if a file with the same name already exists.
            We need to confirm the version: if the version matches, we're good,
            otherwise we need to error out.
        """
    versions, error2 = self.cluster_image_packages_get_rest()
    if self.parameters['package_version'] in versions:
        return True
    if versions:
        self.module.fail_json(msg='Error: another package with the same file name exists: found: %s' % ', '.join(versions))
    self.module.fail_json(msg='Error: ONTAP reported package already exists, but no package found: %s, getting versions: %s' % (error, error2))