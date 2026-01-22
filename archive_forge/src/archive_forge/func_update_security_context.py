from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def update_security_context(self, ref_names, key):
    params = {'kind': 'SecurityContextConstraints', 'api_version': 'security.openshift.io/v1'}
    sccs = self.kubernetes_facts(**params)
    if not sccs['api_found']:
        self.fail_json(msg=sccs['msg'])
    sccs = sccs.get('resources')
    candidates = []
    changed = False
    resource = self.find_resource(kind='SecurityContextConstraints', api_version='security.openshift.io/v1')
    for item in sccs:
        subjects = item.get(key, [])
        retainedSubjects = [x for x in subjects if x not in ref_names]
        if len(subjects) != len(retainedSubjects):
            candidates.append(item['metadata']['name'])
            changed = True
            if not self.check_mode:
                upd_sec_ctx = item
                upd_sec_ctx.update({key: retainedSubjects})
                try:
                    resource.apply(upd_sec_ctx, namespace=None)
                except DynamicApiError as exc:
                    msg = 'Failed to apply object due to: {0}'.format(exc.body)
                    self.fail_json(msg=msg)
    return (candidates, changed)