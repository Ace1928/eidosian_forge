from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic

        sis-policy-create zapi pre-checks the options and fails if it's not supported.
        is-policy-modify pre-checks one of the options, but tries to modify the others even it's not supported. And it will mess up the vsim.
        Do the checks before sending to the zapi.
        This checks applicable for REST modify too.
        