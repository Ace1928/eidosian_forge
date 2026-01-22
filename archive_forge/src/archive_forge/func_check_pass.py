from __future__ import (absolute_import, division, print_function)
from contextlib import contextmanager
import os
import re
import subprocess
import time
import yaml
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.utils.display import Display
from ansible.utils.encrypt import random_password
from ansible.plugins.lookup import LookupBase
from ansible import constants as C
from ansible_collections.community.general.plugins.module_utils._filelock import FileLock
def check_pass(self):
    try:
        self.passoutput = to_text(check_output2([self.pass_cmd, 'show'] + [self.passname], env=self.env), errors='surrogate_or_strict').splitlines()
        self.password = self.passoutput[0]
        self.passdict = {}
        try:
            values = yaml.safe_load('\n'.join(self.passoutput[1:]))
            for key, item in values.items():
                self.passdict[key] = item
        except (yaml.YAMLError, AttributeError):
            for line in self.passoutput[1:]:
                if ':' in line:
                    name, value = line.split(':', 1)
                    self.passdict[name.strip()] = value.strip()
        if self.backend == 'gopass' or os.path.isfile(os.path.join(self.paramvals['directory'], self.passname + '.gpg')) or (not self.is_real_pass()):
            return True
    except subprocess.CalledProcessError as e:
        if 'not in the password store' not in e.output:
            raise AnsibleError('exit code {0} while running {1}. Error output: {2}'.format(e.returncode, e.cmd, e.output))
    if self.paramvals['missing'] == 'error':
        raise AnsibleError('passwordstore: passname {0} not found and missing=error is set'.format(self.passname))
    elif self.paramvals['missing'] == 'warn':
        display.warning('passwordstore: passname {0} not found'.format(self.passname))
    return False