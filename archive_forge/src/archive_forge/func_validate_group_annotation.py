from __future__ import (absolute_import, division, print_function)
import traceback
from datetime import datetime
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.community.okd.plugins.module_utils.openshift_ldap import (
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def validate_group_annotation(definition, host_ip):
    name = definition['metadata']['name']
    annotate_url = definition['metadata'].get('annotations', {}).get(LDAP_OPENSHIFT_URL_ANNOTATION)
    if host_ip:
        if not annotate_url:
            return "group '{0}' marked as having been synced did not have an '{1}' annotation".format(name, LDAP_OPENSHIFT_URL_ANNOTATION)
        elif annotate_url != host_ip:
            return "group '{0}' was not synchronized from: '{1}'".format(name, host_ip)
    annotate_uid = definition['metadata']['annotations'].get(LDAP_OPENSHIFT_UID_ANNOTATION)
    if not annotate_uid:
        return "group '{0}' marked as having been synced did not have an '{1}' annotation".format(name, LDAP_OPENSHIFT_UID_ANNOTATION)
    return None