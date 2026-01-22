from __future__ import absolute_import, division, print_function
import os
import tempfile
import traceback
import re
import json
import copy
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.kubernetes.core.plugins.module_utils.version import (
from ansible.module_utils.basic import AnsibleModule
def run_helm_command(self, command, fails_on_error=True):
    if not HAS_YAML:
        self.fail_json(msg=missing_required_lib('PyYAML'), exception=YAML_IMP_ERR)
    rc, out, err = self.run_command(command, environ_update=self.env_update)
    if fails_on_error and rc != 0:
        self.fail_json(msg='Failure when executing Helm command. Exited {0}.\nstdout: {1}\nstderr: {2}'.format(rc, out, err), stdout=out, stderr=err, command=command)
    return (rc, out, err)