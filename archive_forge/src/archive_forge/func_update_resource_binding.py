from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def update_resource_binding(self, ref_kind, ref_names, namespaced=False):
    kind = 'ClusterRoleBinding'
    api_version = ('rbac.authorization.k8s.io/v1',)
    if namespaced:
        kind = 'RoleBinding'
    resource = self.find_resource(kind=kind, api_version=api_version, fail=True)
    result = resource.get(name=None, namespace=None).to_dict()
    result = result.get('items') if 'items' in result else [result]
    if len(result) == 0:
        return ([], False)

    def _update_user_group(binding_namespace, subjects):
        users, groups = ([], [])
        for x in subjects:
            if x['kind'] == 'User':
                users.append(x['name'])
            elif x['kind'] == 'Group':
                groups.append(x['name'])
            elif x['kind'] == 'ServiceAccount':
                namespace = binding_namespace
                if x.get('namespace') is not None:
                    namespace = x.get('namespace')
                if namespace is not None:
                    users.append('system:serviceaccount:%s:%s' % (namespace, x['name']))
        return (users, groups)
    candidates = []
    changed = False
    for item in result:
        subjects = item.get('subjects', [])
        retainedSubjects = [x for x in subjects if x['kind'] == ref_kind and x['name'] in ref_names]
        if len(subjects) != len(retainedSubjects):
            updated_binding = item
            updated_binding['subjects'] = retainedSubjects
            binding_namespace = item['metadata'].get('namespace', None)
            updated_binding['userNames'], updated_binding['groupNames'] = _update_user_group(binding_namespace, retainedSubjects)
            candidates.append(binding_namespace + '/' + item['metadata']['name'] if binding_namespace else item['metadata']['name'])
            changed = True
            if not self.check_mode:
                try:
                    resource.apply(updated_binding, namespace=binding_namespace)
                except DynamicApiError as exc:
                    msg = 'Failed to apply object due to: {0}'.format(exc.body)
                    self.fail_json(msg=msg)
    return (candidates, changed)