from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_volume
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def fail_on_large_size_reduction(self, app_current, desired, provisioned_size):
    """ Error if a reduction of size > 10% is requested.
            Warn for smaller reduction and ignore it, to protect against 'rounding' errors.
        """
    total_size = app_current['total_size']
    desired_size = desired.get('total_size')
    warning = None
    if desired_size is not None:
        details = 'total_size=%d, provisioned=%d, requested=%d' % (total_size, provisioned_size, desired_size)
        if desired_size < total_size:
            reduction = round((total_size - desired_size) * 100.0 / total_size, 1)
            if reduction > 10:
                self.module.fail_json(msg="Error: can't reduce size: %s" % details)
            else:
                warning = 'Ignoring small reduction (%.1f %%) in total size: %s' % (reduction, details)
        elif desired_size > total_size and desired_size < provisioned_size:
            warning = 'Ignoring increase: requested size is too small: %s' % details
    return warning