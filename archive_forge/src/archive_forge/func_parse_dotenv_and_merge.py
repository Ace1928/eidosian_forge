from __future__ import (absolute_import, division, print_function)
import os
import traceback
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def parse_dotenv_and_merge(self, parameters, parameter_file):
    import re
    DOTENV_PARSER = re.compile('(?x)^(\\s*(\\#.*|\\s*|(export\\s+)?(?P<key>[A-z_][A-z0-9_.]*)=(?P<value>.+?)?)\\s*)[\\r\\n]*$')
    path = os.path.normpath(parameter_file)
    if not os.path.exists(path):
        self.fail(msg='Error accessing {0}. Does the file exist?'.format(path))
    try:
        with open(path, 'r') as f:
            multiline = ''
            for line in f.readlines():
                line = line.strip()
                if line.endswith('\\'):
                    multiline += ' '.join(line.rsplit('\\', 1))
                    continue
                if multiline:
                    line = multiline + line
                    multiline = ''
                match = DOTENV_PARSER.search(line)
                if not match:
                    continue
                match = match.groupdict()
                if match.get('key'):
                    if match['key'] in parameters:
                        self.fail_json(msg="Duplicate value for '{0}' detected in parameter file".format(match['key']))
                    parameters[match['key']] = match['value']
    except IOError as exc:
        self.fail(msg='Error loading parameter file: {0}'.format(exc))
    return parameters