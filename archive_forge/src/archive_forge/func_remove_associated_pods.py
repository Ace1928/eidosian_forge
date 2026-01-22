from __future__ import absolute_import, division, print_function
import re  # noqa: F402
from ansible.module_utils.basic import AnsibleModule  # noqa: F402
def remove_associated_pods(self, pods):
    changed = False
    out_all, err_all = ('', '')
    for pod_id in pods:
        rc, out, err = self._command_run([self.executable, 'pod', 'rm', '-f', pod_id])
        if rc != 0:
            self.module.fail_json('Can NOT delete Pod %s' % pod_id)
        else:
            changed = True
            out_all += out
            err_all += err
    return (changed, out_all, err_all)