from __future__ import (absolute_import, division, print_function)
import traceback
from datetime import datetime
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.community.okd.plugins.module_utils.openshift_ldap import (
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def make_openshift_group(self, group_uid, group_name, usernames):
    group = self.get_group_info(name=group_name)
    if not group:
        group = {'apiVersion': 'user.openshift.io/v1', 'kind': 'Group', 'metadata': {'name': group_name, 'labels': {LDAP_OPENSHIFT_HOST_LABEL: self.module.host}, 'annotations': {LDAP_OPENSHIFT_URL_ANNOTATION: self.module.netlocation, LDAP_OPENSHIFT_UID_ANNOTATION: group_uid}}}
    ldaphost_label = group['metadata'].get('labels', {}).get(LDAP_OPENSHIFT_HOST_LABEL)
    if not ldaphost_label or ldaphost_label != self.module.host:
        return (None, 'Group %s: %s label did not match sync host: wanted %s, got %s' % (group_name, LDAP_OPENSHIFT_HOST_LABEL, self.module.host, ldaphost_label))
    ldapurl_annotation = group['metadata'].get('annotations', {}).get(LDAP_OPENSHIFT_URL_ANNOTATION)
    if not ldapurl_annotation or ldapurl_annotation != self.module.netlocation:
        return (None, 'Group %s: %s annotation did not match sync host: wanted %s, got %s' % (group_name, LDAP_OPENSHIFT_URL_ANNOTATION, self.module.netlocation, ldapurl_annotation))
    ldapuid_annotation = group['metadata'].get('annotations', {}).get(LDAP_OPENSHIFT_UID_ANNOTATION)
    if not ldapuid_annotation or ldapuid_annotation != group_uid:
        return (None, 'Group %s: %s annotation did not match LDAP UID: wanted %s, got %s' % (group_name, LDAP_OPENSHIFT_UID_ANNOTATION, group_uid, ldapuid_annotation))
    group['users'] = usernames
    group['metadata']['annotations'][LDAP_OPENSHIFT_SYNCTIME_ANNOTATION] = datetime.now().isoformat()
    return (group, None)