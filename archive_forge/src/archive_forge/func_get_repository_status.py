from __future__ import absolute_import, division, print_function
import traceback
import copy
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.kubernetes.core.plugins.module_utils.helm import (
from ansible_collections.kubernetes.core.plugins.module_utils.helm_args_common import (
def get_repository_status(module, repository_name):
    list_command = module.get_helm_binary() + ' repo list --output=yaml'
    rc, out, err = module.run_helm_command(list_command, fails_on_error=False)
    if rc == 1 and 'no repositories to show' in err:
        return None
    elif rc != 0:
        module.fail_json(msg='Failure when executing Helm command. Exited {0}.\nstdout: {1}\nstderr: {2}'.format(rc, out, err), command=list_command)
    return get_repository(yaml.safe_load(out), repository_name)